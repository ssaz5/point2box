
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image, ImageDraw 


from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")




def get_scenes_from_dir(directory, split='train', keyword='img1'):
    scene_list = []
    for r,d,f in os.walk(directory):
        if (split in r) and keyword in d :
            scene_list.append(r)
                         
    return scene_list


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# def accimage_loader(path):
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


def default_loader(path):
#     from torchvision import get_image_backend
#     if get_image_backend() == 'accimage':
#         return accimage_loader(path)
#     else:
    return pil_loader(path)

def make_dataset_from_scenes(fnames):
    
    images = []
    gt = np.empty((0,9))
    frame_no = 0
    f_ids = np.array([])
    

        
    
    for i in fnames:
        temp = np.loadtxt(i+'/gt/gt.txt', delimiter=',')
        if '2017' in i:
            keep_indices = np.where(([j in [1,2] for j in temp[:,-2]]))[0]
            temp = temp[keep_indices,:9]
        else:
            temp = temp[:,:9]
        f_ids = np.append(f_ids,temp[:,0]+frame_no)
        
        gt = np.append(gt,temp, axis =0)
        
        
        temp = os.listdir(i+'/img1')
        temp.sort()
        images += [i+'/img1/'+j for j in temp]
        frame_no += len(temp)
    
    f_ids = f_ids.astype(np.int)

    return images, gt, f_ids-1

class MOTDataset_BBwise(Dataset):
    """Point to Box Dataset"""

    def __init__(self, fnames, loader = default_loader, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.fnames = fnames
        self.images, self.gt, self.f_ids = make_dataset_from_scenes(self.fnames)
        self.loader = loader
        self.transform = transform
        self.pt = 0
        self.length = len(self.gt)

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.loader(self.images[self.f_ids[idx]])
        box = self.gt[idx, 2:6]
        

#         sample1 = ImageDraw.Draw(sample)
#         sample1.rectangle(list(box[:2])+list(box[2:]+box[:2]))
        sample1 = sample.crop(list(box[:2])+list(box[2:]+box[:2])) 
    
        if self.transform:
            sample = self.transform(sample)

        return sample1, box
    
    
    def get_next(self):
        self.pt +=1
        return self.__getitem__(self.pt )

    def get_random(self):
        idx = np.random.randint(0,self.length)
        return self.__getitem__(idx), idx