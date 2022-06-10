from locale import normalize
import torch

def model_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

class NLB(torch.nn.Module):
    def __init__(self, in_ch, relu_a=0.01):
        self.inter_ch = torch.div(in_ch, 2, rounding_mode='floor').item()
        super().__init__()
        self.theta_layer = torch.nn.Conv2d(in_channels=in_ch, out_channels=self.inter_ch, \
                            kernel_size=1, padding=0)
        self.phi_layer   = torch.nn.Conv2d(in_channels=in_ch, out_channels=self.inter_ch, \
                            kernel_size=1, padding=0)
        self.g_layer     = torch.nn.Conv2d(in_channels=in_ch, out_channels=self.inter_ch, \
                            kernel_size=1, padding=0)
        self.atten_act   = torch.nn.Softmax(dim=-1)
        self.out_cnn     = torch.nn.Conv2d(in_channels=self.inter_ch, out_channels=in_ch, \
                            kernel_size=1, padding=0)
        
    def forward(self, x):
        mbsz, _, h, w = x.size()
    
        theta = self.theta_layer(x).view(mbsz, self.inter_ch, -1).permute(0, 2, 1)
        phi   = self.phi_layer(x).view(mbsz, self.inter_ch, -1)
        g     = self.g_layer(x).view(mbsz, self.inter_ch, -1).permute(0, 2, 1)
        
        theta_phi = self.atten_act(torch.matmul(theta, phi))
        
        theta_phi_g = torch.matmul(theta_phi, g).permute(0, 2, 1).view(mbsz, self.inter_ch, h, w)
        
        _out_tmp = self.out_cnn(theta_phi_g)
        _out_tmp = torch.add(_out_tmp, x)
   
        return _out_tmp


class BraggNN(torch.nn.Module):
    def __init__(self, imgsz, fcsz=(64, 32, 16, 8)):
        super().__init__()
        self.cnn_ops = []
        cnn_out_chs = (64, 32, 8)
        cnn_in_chs  = (1, ) + cnn_out_chs[:-1]
        fsz = imgsz
        for ic, oc, in zip(cnn_in_chs, cnn_out_chs):
            self.cnn_ops += [
                            torch.nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=3, \
                                stride=1, padding=0),
                            torch.nn.LeakyReLU(negative_slope=0.01),
            ]
            fsz -= 2
        self.nlb = NLB(in_ch=cnn_out_chs[0])
        self.dense_ops = []
        dense_in_chs  = (fsz * fsz * cnn_out_chs[-1], ) + fcsz[:-1]
        for ic, oc in zip(dense_in_chs, fcsz):
            self.dense_ops += [
                            torch.nn.Linear(ic, oc),
                            torch.nn.LeakyReLU(negative_slope=0.01),
            ]
        # output layer
        self.dense_ops += [torch.nn.Linear(fcsz[-1], 2), ]
        
        self.cnn_layers   = torch.nn.Sequential(*self.cnn_ops)
        self.dense_layers = torch.nn.Sequential(*self.dense_ops)

    def normalize_input(self,x):
        mbsz, nchannel, h, w = x.size()
        
        # Normalize the input tensor
        # Make a 2D tensor of dimensions mbsz, nchannel*h*w
        x = x.view(mbsz,-1)
        # Normalize now
        xmin = x.min(1,keepdim=True)[0]
        x -= xmin
        x = torch.div(x, x.max(1,keepdim=True)[0])
        # Make the tensor back to its input shape
        return x.view(mbsz, nchannel, h, w)

    def forward(self, x):
        _out = self.normalize_input(x)
        for layer in self.cnn_layers[:1]:
            _out = layer(_out)

        _out = self.nlb(_out)

        for layer in self.cnn_layers[1:]:
            _out = layer(_out)
        
        _out = _out.flatten(start_dim=1)
        for layer in self.dense_layers:
            _out = layer(_out)
            
        return _out

class DataPreproccessingBlock(torch.nn.Module):
    def __init__( self, out_img_sz, max_random_shift=0, in_img_sz=-1 ) -> None:
        super().__init__()
        self.torch_devs = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.out_img_sz = out_img_sz
        self.out_img_sz_half = self.out_img_sz//2
        self.shift_max = max_random_shift

        # Deal with input image now
        self.in_img_sz = -1
        self.sliced_indices = None
        if in_img_sz > 0:
            self.initializeInputImageSettings(in_img_sz)
        # how to do store index of sliced image from original image
        # Let (fy,fx) be the arbitary start location of the sliced frame wrt to original image of size sz_in x sz_in
        # Let index_matrix[i][j] of size sz_out x sz_out store the flatted index of input image corresponding to i,j pixel of sliced image
        # The following relation hold true
        # index_matrix[i][j] = (fy+j)*sz_in + (fx+i)
        #                    = (fy*sz_in + fx )    +    ( j*sz_in + i )
        #                     Frame dependent part    Pixel dependent part
        # Frame dependent part: Constant offset depending only on the start of frame
        # Pixel dependent part: Independent of frame location ( can be stored beforehand as a global variable )
        # Store the pixel dependent part here unrolled into a 1D column vector:

    def initializeInputImageSettings(self, inp_img_sz):
        # Store the input size first
        self.in_img_sz = inp_img_sz
        # Assuming that patch of size (out_sz x out_sz) is sliced from (in_sz x in_sz) at location (0,0)
        # sliced_indices now store the flatted indices of input image that are part of the output image
        # It is a column vector of total length: out_sz x out_sz 
        self.sliced_indices = torch.zeros((1,self.out_img_sz*self.out_img_sz), dtype=torch.long).to(self.torch_devs)
        for yi in range(self.out_img_sz):
            yoffset = yi*self.out_img_sz
            for xi in range(self.out_img_sz):
                self.sliced_indices[0,yoffset+xi] = yi*self.in_img_sz+xi

    def normalize_patch(self,x):
        mbsz, nchannel, h, w = x.size()
        
        # Normalize the input tensor
        # Make a 2D tensor of dimensions mbsz, nchannel*h*w
        x = x.view(mbsz,-1)
        # Normalize now
        xmin = x.min(1,keepdim=True)[0]
        x -= xmin
        x = torch.div(x, x.max(1,keepdim=True)[0])
        # Make the tensor back to its input shape
        return x.view(mbsz, nchannel, h, w)

    def forward(self, inp_patch, label_loc):
        #label loc is the peak location in (xloc, yloc) or ( column_index, row_index )

        # check if the input image settings are initialized
        # if not, then change the size
        if self.in_img_sz <= 0 :
            self.initializeInputImageSettings(nr)

        # At this point, a valid input image size is set. Next check if we need to clip the patch
        if self.in_img_sz > self.out_img_sz:
            nbatch, nch, nr, nc = inp_patch.size()
            # generate random shifts (x,y) for entire minibatch
            rnd_shift = torch.randint( -self.shift_max, self.shift_max+1, (nbatch, 2), dtype=torch.float32 ).to(self.torch_devs)

            # shift peak location
            # If peak location is not provided, then clip around the center of the input image
            if label_loc is None:
                rnd_shift += self.in_img_sz//2
            else:
                rnd_shift +=label_loc
            # get the frame start location
            frame_start  = rnd_shift.int()
            frame_start -= self.out_img_sz_half
            # # make sure that the frame start is not negative
            # frame_start = torch.maximum(frame_start, torch.zeros(frame_start.size()))

            #First calculate the batch index offset
            batch_index_offset = torch.arange(0, nbatch*self.in_img_sz**2, self.in_img_sz**2).view(nbatch,-1).to(self.torch_devs)
            # calculate the frame dependent offset of index
            # (fy*sz_in + fx ) or in this case : frame_start[:,0] + sz_in*frame_start[:,1]
            frame_index_offset = torch.add(frame_start[:,0], frame_start[:,1], alpha=self.in_img_sz).view(nbatch,-1)
            # Add batch index offset to the frame offset
            frame_index_offset += batch_index_offset

            # indexes of a single dimension vector
            indices = torch.add(frame_index_offset,self.sliced_indices)
            # unwrap indices to a single vector
            indices = indices.view(-1).type(torch.long)
            # flatten the bigger image, select the indices for smaller patch and reshape the tensor
            inp_patch = inp_patch.view(-1)[indices].view((nbatch,nch,self.out_img_sz, self.out_img_sz))

            # get the new normalized peak location for this small image patch
            label_loc -= frame_start
            label_loc /= self.out_img_sz

        else:
            # No clipping is required, but simply normalize the peak location between 0,1
            label_loc /=self.out_img_sz

        return inp_patch, label_loc.type(torch.float32)

class TrainingModelWithLoss(torch.nn.Module):
    def __init__(self, model, in_sz: int, out_sz: int, max_shift: int):
        super().__init__()
        self.model = model
        self.augmentation = None
        self.loss = torch.nn.MSELoss()
        self.augmentation = DataPreproccessingBlock(out_sz, max_random_shift=max_shift, in_img_sz= in_sz)

    def forward(self, input_patch, loss_inputs=None):
        if self.augmentation:
            aug_patch, aug_label = self.augmentation(input_patch, loss_inputs)
        else:
            aug_patch = input_patch
            aug_label = loss_inputs
        
        output = self.model(aug_patch)
        if loss_inputs is None:
            return output
        else:
            loss = self.loss(output, aug_label)
            return output, loss