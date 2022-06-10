from ast import arg
import pickle
pickle.DEFAULT_PROTOCOL=4

import torch, argparse, os, time, sys, shutil, logging
from model import model_init, BraggNN, TrainingModelWithLoss, DataPreproccessingBlock
from torch.utils.data import DataLoader
from dataset import BraggDatasetOptimized
from tqdm import tqdm
import numpy as np

class BraggNN_Trainer(object):
    def __init__( self, device_name ) -> None:
        self.device_name = device_name
        self.dl_train = None
        self.training_model = None
        self.dl_val = None
        self.images_trained = 0
        self.images_validated = 0

        # Total time taken inside compute iterations
        self.train_time = 0
        self.valid_time = 0
        print("BraggNN training running on: ", self.device_name)
    
    def getDeviceName(self) -> str:
        return self.device_name

class BraggNN_IPU( BraggNN_Trainer ):
    def __init__(self, model, args, train_dataset=None, val_dataset=None):
        super().__init__("IPU")

        assert ((model is not None) and (train_dataset is not None)), " BraggNN_IPU: Model and Training dataset must not be NONE" 

        # Dynamically Import poptorch module for IPU
        self.poptorch = __import__('poptorch')

        self.model_with_loss = None
        self.inference_model = None

        # Total time taken inside compute iterations
        self.train_time = 0
        self.valid_time = 0

        # ------------------------ #
        # Setup the training model
        # ------------------------ #
        # First Create model options from PopTorch
        train_model_opts = self.getModelOptions( cache = (not args.no_cache), device_iter = args.device_iter,
                                                 replicas = args.num_train_replica, benchmark = args.benchmark )

        self.dl_train = self.getDataloader(train_dataset, train_model_opts, args)
        # Get the sample image for the input dataset
        train_img, train_lbl = next(iter(self.dl_train))
        # Get input image size
        _, _, in_img_rows, _ = train_img.size()
        train_model_with_loss = TrainingModelWithLoss(model, in_img_rows, args.psz, args.aug)

        optimizer = self.poptorch.optim.Adam(model.parameters(), lr=args.lr) 
        self.training_model = self.poptorch.trainingModel(train_model_with_loss, options=train_model_opts, optimizer=optimizer)

        # Compile the Training the first
        if not self.training_model.isCompiled():
            time2compile = time.perf_counter()
            print("Compiling Training Model")
            self.training_model.compile( train_img, train_lbl )
            time2compile = time.perf_counter()-time2compile
            print("Training model compile time: {:.3f} s".format(time2compile))

        # ------------------------------------- #
        # Setup the validation/inference model
        # ------------------------------------- #
        if val_dataset is not None:
            val_model_opts = self.getModelOptions( cache = (not args.no_cache), replicas = args.num_infer_replica,
                                                   benchmark = args.benchmark )


            self.dl_valid = self.getDataloader(val_dataset, val_model_opts, args)
            # Get the sample image for the input dataset
            valid_img, valid_lbl = next(iter(self.dl_valid))

            # Create an inference model to get validation loss
            # This will run on different IPUs than training model
            # Note: Inference model is passing a same model to keep the weights in sync
            self.inference_model = self.poptorch.inferenceModel(train_model_with_loss, options=val_model_opts)

            if not self.inference_model.isCompiled():
                time2compile = time.perf_counter()
                print("Compiling Inference Model")
                self.inference_model.compile(valid_img, valid_lbl)
                time2compile = time.perf_counter()-time2compile
                print("Inference model compile time: {:.3f} s".format(time2compile))

    def getDataloader(self, ds, opts, args):
        async_options = { "sharing_strategy": self.poptorch.SharingStrategy.SharedMemory,
                        "early_preload":     True, "buffer_size": args.num_threads,
                        "load_indefinitely": True, "miss_sleep_time_in_ms": 0 }

        return self.poptorch.DataLoader( options=opts, 
                                    dataset=ds,
                                    batch_size = args.mbsz,
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=args.num_threads,
                                    persistent_workers=True,
                                    mode = self.poptorch.DataLoaderMode.Async,
                                    async_options=async_options
                                  )

    def getModelOptions(self, cache = True, device_iter = 1, replicas = 1, benchmark = False):
        model_opts = self.poptorch.Options()
        model_opts.deviceIterations(device_iter)
        model_opts.replicationFactor(replicas)
        if cache:
            model_opts.enableExecutableCaching("./cache")
        if benchmark:
            model_opts.enableSyntheticData(True)
        return model_opts

    def run_training_epoch(self):
        images_trained_this_epoch = 0
        images_validated_this_epoch = 0
        train_loss = 0
        valid_loss = 0
        #Start the epoch
        epoch_tick = time.perf_counter()
        for X_mb, y_mb in self.dl_train:
            # Start iteration timer
            it_comp_tick = time.perf_counter()

            # Run a training iteration
            _, loss = self.training_model(X_mb, y_mb)
            # Record loss of each batch
            train_loss += loss.item()

            # Stop iteration timer
            self.train_time += (time.perf_counter()-it_comp_tick)
            #Get shape of the input tensor to calculate number of images processed
            batch_shape = list(X_mb.size())
            
            images_trained_this_epoch += batch_shape[0]
        train_loss /= len(self.dl_train)

        #update the weights on the torch model on host
        self.training_model.copyWeightsToHost()
        
        if self.dl_valid is not None:
            valid_loss = 0
            with torch.no_grad():
                for data, labels in self.dl_valid:
                    t_start = time.perf_counter()
                    _, loss = self.inference_model(data, labels)
                    self.valid_time += time.perf_counter()-t_start
                    valid_loss += loss.item()
                    images_validated_this_epoch += list(data.size())[0]

            valid_loss /= len(self.dl_valid)
        time2epoch = time.perf_counter()-epoch_tick

        self.images_trained += images_trained_this_epoch
        self.images_validated += images_validated_this_epoch
        epoch_throughput = images_trained_this_epoch/time2epoch

        return train_loss, valid_loss, epoch_throughput


class BraggNN_GPU( BraggNN_Trainer ):
    def __init__(self, model, args, train_dataset=None, val_dataset=None):
        super().__init__("GPU" if torch.cuda.is_available() else "CPU" )
        assert ((model is not None) and (train_dataset is not None)), " BraggNN_IPU: Model and Training dataset must not be NONE" 

        # ------------------------ #
        # Setup the training model
        # ------------------------ #
        self.training_model = model
        self.torch_devs = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dl_train = DataLoader( dataset=train_dataset, batch_size=args.mbsz, shuffle=True,\
                                    num_workers=args.num_threads, prefetch_factor=args.mbsz, drop_last=True, pin_memory=True )
        # Get the sample image for data augmentation block
        train_img, train_lbl = next(iter(self.dl_train))
        # Get input image size
        _, _, in_img_rows, _ = train_img.size()
        self.model_preprocess = DataPreproccessingBlock(args.psz, max_random_shift=args.aug, in_img_sz=in_img_rows)
        # Transfer to model and preprocessing to GPUs
        if torch.cuda.is_available():
            self.training_model.to(self.torch_devs)
            self.model_preprocess.to(self.torch_devs)
        
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 

        # ------------------------------------- #
        # Setup the validation/inference model
        # ------------------------------------- #
        if val_dataset is not None:
            self.dl_valid = DataLoader( dataset=val_dataset, batch_size=args.mbsz, shuffle=True,\
                                      num_workers=args.num_threads, prefetch_factor=args.mbsz, drop_last=True, pin_memory=True )

    def run_training_epoch(self):
        images_trained_this_epoch = 0
        images_validated_this_epoch = 0
        train_loss = 0
        valid_loss = 0
        #Start the epoch
        epoch_tick = time.perf_counter()
        for X_mb, y_mb in self.dl_train:
            # Start iteration timer
            it_comp_tick = time.perf_counter()

            # Run a training iteration
            # Run a training iteration
            self.optimizer.zero_grad()
            aug_img, aug_label = self.model_preprocess(X_mb.to(self.torch_devs), y_mb.to(self.torch_devs))
            pred = self.training_model.forward(aug_img)
            loss = self.criterion(pred, aug_label)
            loss.backward()
            self.optimizer.step() 
            # Record loss of each batch
            train_loss += loss.item()

            # Stop iteration timer
            self.train_time += (time.perf_counter()-it_comp_tick)
            #Get shape of the input tensor to calculate number of images processed
            batch_shape = list(X_mb.size())
            
            images_trained_this_epoch += batch_shape[0]
        train_loss /= len(self.dl_train)
       
        if self.dl_valid is not None:
            valid_loss = 0
            with torch.no_grad():
                for data, labels in self.dl_valid:
                    t_start = time.perf_counter()
                    aug_img, aug_label = self.model_preprocess(data.to(self.torch_devs), labels.to(self.torch_devs))
                    pred_val = self.training_model(aug_img)
                    loss = self.criterion(pred_val, aug_label)
                    self.valid_time += time.perf_counter()-t_start
                    valid_loss += loss.item()
                    images_validated_this_epoch += list(data.size())[0]

            valid_loss /= len(self.dl_valid)
        time2epoch = time.perf_counter()-epoch_tick

        self.images_trained += images_trained_this_epoch
        self.images_validated += images_validated_this_epoch
        epoch_throughput = images_trained_this_epoch/time2epoch

        return train_loss, valid_loss, epoch_throughput

def getBraggTrainer(model, args, train_dataset=None, val_dataset=None):
    if args.device=='ipu':
        return BraggNN_IPU(model, args, train_dataset=train_dataset, val_dataset=val_dataset)
    else:
        return BraggNN_GPU(model, args, train_dataset=train_dataset, val_dataset=val_dataset)

def execute(args):

    itr_out_dir = "BraggNN_"+args.expName
    if os.path.isdir(itr_out_dir):
        shutil.rmtree(itr_out_dir)
    os.mkdir(itr_out_dir) # to save temp output

    logging.basicConfig(filename=os.path.join(itr_out_dir, 'BraggNN.log'), level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    #####################################################
    #          LOAD THE DATASETS AND DATALOADERS        #
    #####################################################
    # Regarding dataset loading:
    # Loading the data from disk to memory can be slow based on the size of dataset
    # The preprocessing of data is a single-threaded operation.
    # To speed up dataset initialization, use multithreading

    # Now the dataset contains big images ( 2048 x 2048 ) containing multiple Bragg peaks
    # BraggNN model accepts small patch size (psz) around a Bragg peak e.g 11 x 11 by default
    # In this code, images of size ( psz + pad ) x ( psz + pad ) are extracted from bigger 2048 x 2048 images
    # Here, pad depends upon the maximum random shift around Bragg peak needed for data-augmentation
    # Data augmentation is done on IPU, therefore making dataloading on host side efficient

    # Extract the images from frames with the following info:
    # 1) Size: ( psz + 2*(rnd_shift+1) ) x ( psz + 2*(rnd_shift+1) )
    # 2) In (rand_shift+1), extra 1 pixels are added for safety

    ds_img_sz = args.psz+2*(args.aug+1)
    dl_time_start = time.perf_counter()
    ds_train = BraggDatasetOptimized( dataset=args.dataset, padded_img_sz=ds_img_sz, psz=args.psz, use='train', train_frac=0.8 )
    ds_valid = BraggDatasetOptimized( dataset=args.dataset, padded_img_sz=ds_img_sz, psz=args.psz, use='validation', train_frac=0.8 )
    dl_time_end = time.perf_counter()

    # Print dataset statistics:
    logging.info("Dataset Statistics: ")
    logging.info("  Loading time:  %.3f seconds"      % (dl_time_end-dl_time_start))
    logging.info("  Samples in training datset: %d"   % ( len(ds_train)))
    logging.info("  Samples in validation datset: %d" % ( len(ds_valid)))

    # Set up the model for training
    model = BraggNN(imgsz=args.psz, fcsz=args.fcsz)

    _ = model.apply(model_init) # init model weights and bias
    model_trainer = getBraggTrainer(model, args, train_dataset=ds_train, val_dataset=ds_valid)


    # Store epoch id, time, mean training loss, mean validation, and average throughtput for epochs
    loss_trend = np.zeros((args.maxep,5))

    # set the epoch bar
    prog_bar = tqdm(range(1,args.maxep+1))

    train_time_start = time.perf_counter()
    
    logging.info("Training BraggNN on {0}".format(model_trainer.getDeviceName()))
    for epoch in prog_bar:
        epoch_tick = time.perf_counter()
        loss_train, loss_valid, throughput = model_trainer.run_training_epoch()
        curr_time = time.perf_counter()-train_time_start
        loss_trend[epoch-1,:] = [epoch, curr_time, loss_train, loss_valid, throughput]

        #Update the progress bar
        prog_bar.set_description(" Training Loss:{:0.6f}, Validation Loss:{:0.6f} Img/s: {:d}".format(loss_train, loss_valid, int(throughput)))
        torch.save(model.state_dict(), "%s/mdl-it%05d.pth" % (itr_out_dir, epoch))

    train_time_end = time.perf_counter()
    tot_time_training = train_time_end-train_time_start
    total_images_processed = model_trainer.images_trained + model_trainer.images_validated
    avg_throughput         = total_images_processed/tot_time_training

    logging.info("\n\n### TRAINING PERFORMANCE STATS ###")
    logging.info("Training Device: {0}".format(model_trainer.getDeviceName()))
    logging.info("Number of epochs: {0}".format(args.maxep))
    logging.info("Minibatch size: {0}".format(args.mbsz))
    if (model_trainer.getDeviceName()=="IPU"):
        logging.info("Device iteration count: {0}".format(args.device_iter))
    logging.info("Total images proccesed: {0}".format(total_images_processed))
    logging.info("Total time to train: {:3f}".format(tot_time_training))
    logging.info("Total training time on device: {:3f}".format(model_trainer.train_time))
    logging.info("Total validation time on device: {:3f}".format(model_trainer.valid_time))
    logging.info("Training efficiency: {:2f} %".format(100*(model_trainer.train_time+model_trainer.valid_time)/tot_time_training))
    logging.info("Average training throughput: {:1f} img/s".format(avg_throughput))
    logging.info("Average loss of last iteration: {0}".format(loss_trend[-1,2]))
    logging.info("#########################")
    
    np.savetxt( os.path.join(itr_out_dir,"loss_iter.dat"), loss_trend)

    # Copy the final weights of the trained model to the host
    torch.save(model.state_dict(), "%s/final_model.pth" % (itr_out_dir))
