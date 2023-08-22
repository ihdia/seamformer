'''
Pytorch Dataloader for Scribble & Binarisation Training

Note : Refer to the format mentioned in 
the README to understand the structure of the 
dataset.

'''

import os
import sys 
import torch
import torch.utils.data as D
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
import albumentations as A
import random


class ScribbleDataset(D.Dataset):
    def __init__(self,config,base_dir,file_label,set,augmentation=True,flipped=False):
        self.config = config
        self.base_dir = base_dir
        # File where all the images paths are stored.
        self.file_label = file_label
        self.set = set
        self.augmentation = augmentation
        self.flipped = flipped
        self.indiceList = list(range(0,len(self.file_label)))
        self.patchSize = 256

        # Rotation Transformation
        self.rotatetransform = A.Compose([A.SafeRotate(limit=20,interpolation=1,border_mode = cv2.BORDER_WRAP,value=None,mask_value=None,always_apply=False, p=0.75)])
        # Spatial Transformation 
        self.spatialtransform = A.Compose([A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.CoarseDropout(max_holes=4, max_height=64,max_width=64,min_holes=1,min_height=24, min_width=24, fill_value=0, mask_fill_value=0,always_apply=False,p=0.6)
                ])
        # Quality Transformation 
        self.qualitytransform =  A.Compose([
            A.AdvancedBlur(blur_limit=(3, 7), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0), rotate_limit=90, beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1), always_apply=False, p=0.5),
            A.RandomFog (fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=0.6),
            A.RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.6),
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.8)]
        )
    
    def __len__(self):
        return len(self.file_label)
    
    def __getitem__(self,index):
        img_name = self.file_label[index]
        idx, deg_img, gt_img = self.readImages(img_name) 
        if deg_img is None or gt_img is None :
            self.weightList[int(index)]=0.0
            flag=False
            while(flag is False):
                newindex = random.choices(self.indiceList,k=1)[0]
                img_name = self.file_label[newindex]
                idx, deg_img, gt_img = self.readImages(img_name)
                if deg_img is not None :
                    flag=True
                    break 
                else:
                    continue
        return idx, deg_img, gt_img

    def readImages(self,file_name):
        # Refer documentation for more details of how dataset folder should be structured.
        if self.config['train_scribble']:
            imageURL = self.base_dir+'{}_{}_patches'.format(self.config['dataset_code'],self.set)+'/images/'+file_name
            gtURL = self.base_dir+'{}_{}_patches'.format(self.config['dataset_code'],self.set)+'/scribbleMap/'+file_name.replace('im_','sm_')
        if self.config['train_binary']:
            imageURL = self.base_dir+'{}_{}_patches'.format(self.config['dataset_code'],self.set)+'/images/'+file_name
            gtURL = self.base_dir+'{}_{}_patches'.format(self.config['dataset_code'],self.set)+'/binaryImages/'+file_name.replace('im_','bm_')
        
        if not (os.path.exists(imageURL) or not(os.path.exists(gtURL))):
            print("Image {} DOES NOT exists at the specified location.".format(imageURL))
            return None,None,None
        
        deg_img = cv2.imread(imageURL)
        gt_img = cv2.imread(gtURL)

        # Note : Inverting pixel intensities such that text(foreground) is white and background is black. 
        # Useful for weighted BCE loss
        if self.config['train_binary']:
            gt_img = cv2.imread(gtURL)

        H,W,C = deg_img.shape
        if H<self.patchSize or W<self.patchSize:
            dim = (self.patchSize,self.patchSize)
            deg_img  = cv2.resize(deg_img, dim, interpolation = cv2.INTER_AREA)
            gt_img  = cv2.resize(gt_img, dim, interpolation = cv2.INTER_AREA)

        # Not gonna apply rotation on all patches 
        if self.augmentation:
            augProb=random.random()
            if augProb>=0.5 : 
                stransformed = self.spatialtransform(image = deg_img , mask =gt_img)
                deg_img = stransformed['image']
                gt_img = stransformed['mask']
            else:
                qtransformed = self.qualitytransform(image = deg_img ,mask= gt_img)
                deg_img = qtransformed['image']
                gt_img = qtransformed['mask']

        # Normalize 
        deg_img = (np.array(deg_img)/255).astype('float32')
        gt_img = (np.array(gt_img)/255).astype('float32')

        # Mean Normalization of Image & GT
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        out_deg_img = np.zeros([3, *deg_img.shape[:-1]])
        out_gt_img = np.zeros([3, *gt_img.shape[:-1]])
        for i in range(3):
            out_deg_img[i] = (deg_img[:,:,i] - mean[i]) / std[i]
            out_gt_img[i] = gt_img[:,:,i]    

        return file_name, out_deg_img, out_gt_img

'''
Load Entire Dataset 
'''

def loadDatasets(config,base_dir,flipped=False):
    # Load from the location
    data_tr = list(os.listdir(base_dir+'{}_train_patches/images/'.format(config['dataset_code'])))
    data_va = list(os.listdir(base_dir+'{}_test_patches/images/'.format(config['dataset_code'])))
    print('For Dataset : {} Train List : {} Test List : {}'.format(config['dataset_code'],len(data_tr),len(data_va)))
    try:
        assert len(data_tr)>1 and len(data_va)>1,"Error in listing images!"
    except Exception as exp:
        print('Exiting!-Insufficient Samples : {}'.format(exp))
        sys.exit()

    # Dataset ( Train - Test ) 
    # Note : You can assume your entire validation set will be labelled as 'test'
    data_train = ScribbleDataset(config,base_dir, data_tr, 'train', augmentation=True)
    data_valid = ScribbleDataset(config,base_dir, data_va, 'test', augmentation=False, flipped = flipped)
    return data_train, data_valid


'''
    Transform a batch of data to pytorch tensor
    Args:
        batch [str, np.array, np.array]: a batch of data
    Returns:
        data_index (tensor): the indexes of the source/target pair
        data_in (tensor): the source images (degraded)
        data_out (tensor): the target images (clean gt)
'''

def sort_batch(batch):
    n_batch = len(batch)
    data_index = []
    data_in = []
    data_out = []

    for i in range(0,n_batch):
        idx, img, gt_img = batch[i]
        data_index.append(idx)
        data_in.append(img)
        data_out.append(gt_img)

    data_index = np.array(data_index)
    data_in = np.array(data_in, dtype=np.float32)
    data_out = np.array(data_out, dtype=np.float32)

    data_in = torch.from_numpy(data_in)
    data_out = torch.from_numpy(data_out)

    return data_index, data_in, data_out


"""
Create the 2 data loaders

Args:
    batch_size (int): the batch_size
Returns:
    train_loader (dataloader): train data loader  
    valid_loader (dataloader): valid data loader
    test_loader (dataloader): test data loader
"""

def all_data_loader(config,batch_size=4):
    base_dir =config["data_path"]
    data_train, data_valid = loadDatasets(config,base_dir)
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=sort_batch, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid, collate_fn=sort_batch, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, valid_loader 

