import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import random
import logging
from skimage.feature import peak_local_max
from skimage import measure
import os, sys

import pickle
pickle.DEFAULT_PROTOCOL=4
import time

def clean_patch(p, center):
    w, h = p.shape
    cc = measure.label(p > 0)
    if cc.max() == 1:
        return p

    # logging.warn(f"{cc.max()} peaks located in a patch")
    lmin = np.inf
    cc_lmin = None
    for _c in range(1, cc.max()+1):
        lmax = peak_local_max(p * (cc==_c), min_distance=1)
        if lmax.shape[0] == 0:continue # single pixel component
        lc = lmax.mean(axis=0)
        dist = ((lc - center)**2).sum()
        if dist < lmin:
            cc_lmin = _c
            lmin = dist
    return p * (cc == cc_lmin)

class BraggNNDataset(Dataset):
    def __init__(self, dataset='./dataset', psz=11, rnd_shift=0, use='train', train_frac=0.8):
        self.psz = psz 
        self.rnd_shift = rnd_shift
        
        with h5py.File( os.path.join(dataset, 'peaks-exp4train-psz%d.hdf5' % psz), "r") as h5fd: 
            if use == 'train':
                sti, edi = 0, int(train_frac * h5fd['peak_fidx'].shape[0])
            elif use == 'validation':
                sti, edi = int(train_frac * h5fd['peak_fidx'].shape[0]), None
            else:
                logging.error(f"unsupported use: {use}. This class is written for building either training or validation set")

            mask = h5fd['npeaks'][sti:edi] == 1 # use only single-peak patches
            mask = mask & ((h5fd['deviations'][sti:edi] >= 0) & (h5fd['deviations'][sti:edi] < 1))

            self.peak_fidx= h5fd['peak_fidx'][sti:edi][mask]
            self.peak_row = h5fd['peak_row'][sti:edi][mask]
            self.peak_col = h5fd['peak_col'][sti:edi][mask]

        self.fidx_base = self.peak_fidx.min()
        # only loaded frames that will be used
        
        with h5py.File(os.path.join(dataset,'frames-exp4train.hdf5'), 'r') as h5fd: 
            self.frames = h5fd['frames'][self.peak_fidx.min():self.peak_fidx.max()+1]
        self.len = self.peak_fidx.shape[0]

    def __getitem__(self, idx):
        _frame = self.frames[self.peak_fidx[idx] - self.fidx_base]
        if self.rnd_shift > 0:
            row_shift = np.random.randint(-self.rnd_shift, self.rnd_shift+1)
            col_shift = np.random.randint(-self.rnd_shift, self.rnd_shift+1)
        else:
            row_shift, col_shift = 0, 0
        prow_rnd  = int(self.peak_row[idx]) + row_shift
        pcol_rnd  = int(self.peak_col[idx]) + col_shift

        row_base = max(0, prow_rnd-self.psz//2)
        col_base = max(0, pcol_rnd-self.psz//2 )

        crop_img = _frame[row_base:(prow_rnd + self.psz//2 + self.psz%2), \
                          col_base:(pcol_rnd + self.psz//2  + self.psz%2)]
        # if((crop_img > 0).sum() == 1): continue # ignore single non-zero peak
        if crop_img.size != self.psz ** 2:
            c_pad_l = (self.psz - crop_img.shape[1]) // 2
            c_pad_r = self.psz - c_pad_l - crop_img.shape[1]

            r_pad_t = (self.psz - crop_img.shape[0]) // 2
            r_pad_b = self.psz - r_pad_t - crop_img.shape[0]

            logging.warn(f"sample {idx} touched edge when crop the patch: {crop_img.shape}")
            crop_img = np.pad(crop_img, ((r_pad_t, r_pad_b), (c_pad_l, c_pad_r)), mode='constant')
        else:
            c_pad_l, r_pad_t = 0 ,0

        _center = np.array([self.peak_row[idx] - row_base + r_pad_t, self.peak_col[idx] - col_base + c_pad_l])
        crop_img = clean_patch(crop_img, _center)
        if crop_img.max() != crop_img.min():
            _min, _max = crop_img.min().astype(np.float32), crop_img.max().astype(np.float32)
            feature = (crop_img - _min) / (_max - _min)
        else:
            logging.warn("sample %d has unique intensity sum of %d" % (idx, crop_img.sum()))
            feature = crop_img

        px = (self.peak_col[idx] - col_base + c_pad_l) / self.psz
        py = (self.peak_row[idx] - row_base + r_pad_t) / self.psz

        return feature[np.newaxis], np.array([px, py]).astype(np.float32)

    def __len__(self):
        return self.len

class BraggDatasetLite(Dataset):
    def __init__( self, dataset='./dataset', psz=11, rnd_shift=0, use='train', train_frac=0.8 ):
        super().__init__()
        # ds = BraggNNDataset(psz=psz, rnd_shift=shift, use=use)
        self.psz = psz # Size of image
        self.rnd_shift = rnd_shift # Random shift required for data augmentation
        
        # Extract the information about rows and columns
        with h5py.File( os.path.join(dataset, 'peaks-exp4train-psz%d.hdf5' % psz), "r") as h5fd: 
            if use == 'train':
                sti, edi = 0, int(train_frac * h5fd['peak_fidx'].shape[0])
            elif use == 'validation':
                sti, edi = int(train_frac * h5fd['peak_fidx'].shape[0]), None
            else:
                logging.error(f"unsupported use: {use}. This class is written for building either training or validation set")

            mask = h5fd['npeaks'][sti:edi] == 1 # use only single-peak patches
            mask = mask & ((h5fd['deviations'][sti:edi] >= 0) & (h5fd['deviations'][sti:edi] < 1))

            peak_fidx= h5fd['peak_fidx'][sti:edi][mask]
            peak_row = h5fd['peak_row'][sti:edi][mask]
            peak_col = h5fd['peak_col'][sti:edi][mask]
        
        fidx_base = peak_fidx.min()
        # only loaded frames that will be used
        with h5py.File(os.path.join(dataset,'frames-exp4train.hdf5'), 'r') as h5fd: 
            frames = h5fd['frames'][peak_fidx.min():peak_fidx.max()+1]

        # Length of the dataset
        self.len = peak_fidx.shape[0]

        # Extract the images from frames with the following info:
        # 1) Size: ( psz + 2*(rnd_shift+1) ) x ( psz + 2*(rnd_shift+1) )
        # 2) In (rand_shift+1), extra 1 pixels are added for padding
        
        self.frame_sz = psz+2*(rnd_shift+1)

        # Preallocate arrays for labels and images
        self.peak_loc = np.zeros((self.len,2), dtype=np.float32)
        self.imgs   = np.zeros((self.len, self.frame_sz, self.frame_sz), dtype=np.float32)

        for ii in range(self.len):
            # Fetch current frame
            curr_frame = frames[peak_fidx[ii] - fidx_base]
            (frame_row, frame_col) = curr_frame.shape
            curr_row = int(peak_row[ii])
            row_base = max(0, curr_row-self.frame_sz//2)
            row_max  = min(frame_row, curr_row+self.frame_sz//2+self.frame_sz%2)
            curr_col = int(peak_col[ii])
            col_base = max(0, curr_col-self.frame_sz//2)
            col_max  = min(frame_col, curr_col+self.frame_sz//2+self.frame_sz%2)
            self.peak_loc[ii,0] = peak_row[ii]-row_base
            self.peak_loc[ii,1] = peak_col[ii]-col_base
            self.imgs[ii,...] = curr_frame[row_base:row_max, col_base:col_max]
   
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        curr_frame = self.imgs[idx]
        curr_peak  = self.peak_loc[idx]
        if self.rnd_shift > 0:
            peak_shift = np.random.randint(-self.rnd_shift, self.rnd_shift+1, size=2).astype(np.float32)
        else:
            peak_shift = np.array([0,0]).astype(np.float32)
        peak_shift += self.peak_loc[idx]

        frame_start = np.array([ max(0, int(pi)-self.psz//2) for pi in peak_shift ])
        frame_end  = frame_start+self.psz

        crop_img = curr_frame[ frame_start[0]:frame_end[0], frame_start[1]:frame_end[1] ] 

        peak_center = curr_peak - frame_start
        crop_img = clean_patch(crop_img, peak_center)
        if crop_img.max() != crop_img.min():
            _min, _max = crop_img.min().astype(np.float32), crop_img.max().astype(np.float32)
            feature = (crop_img - _min) / (_max - _min)
        else:
            logging.warn("sample %d has unique intensity sum of %d" % (idx, crop_img.sum()))
            feature = crop_img

        label = (curr_peak-frame_start)/self.psz

        return feature[np.newaxis], np.array([label[1], label[0]]).astype(np.float32)
        
class BraggDatasetLitest(Dataset):
    def __init__( self, dataset='./dataset', psz=11, rnd_shift=0, use='train', train_frac=0.8 ):
        ds = BraggNNDataset(dataset=dataset, psz=psz, rnd_shift=rnd_shift, use=use, train_frac=train_frac)
        self.len = len(ds)
        self.labels = np.zeros((self.len,2), dtype=np.float32)
        self.images = np.zeros((self.len,1,psz, psz), dtype=np.float32)

        for idx, (img,lab) in enumerate(ds):
            self.labels[idx,...] = lab
            self.images[idx,...] = img
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,idx):
        return self.images[idx,...], self.labels[idx,...]

class BraggDatasetOptimized(Dataset):
    def __init__( self, dataset='./dataset', padded_img_sz=11, psz = 11, use='train', train_frac=0.8 ):
        super().__init__()
        self.frame_sz = padded_img_sz # Size of image
        
        # Extract the information about rows and columns
        with h5py.File( os.path.join(dataset, 'peaks-exp4train-psz%d.hdf5' % psz), "r") as h5fd: 
            if use == 'train':
                sti, edi = 0, int(train_frac * h5fd['peak_fidx'].shape[0])
            elif use == 'validation':
                sti, edi = int(train_frac * h5fd['peak_fidx'].shape[0]), None
            else:
                logging.error(f"unsupported use: {use}. This class is written for building either training or validation set")

            mask = h5fd['npeaks'][sti:edi] == 1 # use only single-peak patches
            mask = mask & ((h5fd['deviations'][sti:edi] >= 0) & (h5fd['deviations'][sti:edi] < 1))

            peak_fidx= h5fd['peak_fidx'][sti:edi][mask]
            peak_row = h5fd['peak_row'][sti:edi][mask]
            peak_col = h5fd['peak_col'][sti:edi][mask]
        
        fidx_base = peak_fidx.min()
        # only loaded frames that will be used
        with h5py.File(os.path.join(dataset,'frames-exp4train.hdf5'), 'r') as h5fd: 
            frames = h5fd['frames'][peak_fidx.min():peak_fidx.max()+1]

        # Length of the dataset
        self.len = peak_fidx.shape[0]
        
        # Preallocate arrays for labels and images
        self.peak_loc = np.zeros((self.len,2), dtype=np.float32)
        self.imgs   = np.zeros((self.len, 1, self.frame_sz, self.frame_sz), dtype=np.float32)

        for ii in range(self.len):
            # Fetch current frame
            curr_frame = frames[peak_fidx[ii] - fidx_base]
            (frame_row, frame_col) = curr_frame.shape
            curr_row = int(peak_row[ii])
            row_base = max(0, curr_row-self.frame_sz//2)
            row_max  = min(frame_row, curr_row+self.frame_sz//2+self.frame_sz%2)
            curr_col = int(peak_col[ii])
            col_base = max(0, curr_col-self.frame_sz//2)
            col_max  = min(frame_col, curr_col+self.frame_sz//2+self.frame_sz%2)
            self.peak_loc[ii,0] = peak_row[ii]-row_base
            self.peak_loc[ii,1] = peak_col[ii]-col_base
            self.imgs[ii,0, ...] = clean_patch(curr_frame[row_base:row_max, col_base:col_max], self.peak_loc[ii,:])

        # Flip the order of columns in peak_loc: Row(y), Col(x) --> Col(x), Row(y)
        self.peak_loc = np.fliplr(self.peak_loc)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.imgs[idx,...].copy()), torch.from_numpy(self.peak_loc[idx,...].copy())
