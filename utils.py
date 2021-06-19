from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor
from PIL import Image
import torch
import tqdm
from diskcache import FanoutCache
from torch.utils.data import Dataset
import os
import glob
import scipy.io as sio
from disk import getCache

cache = getCache('ProjectCache')


transform = Compose([ToPILImage(),Resize((227,227)), ToTensor()]) 


# @cache.memoize(typed=True, tag='stride')
def generate_stride_set(video_array, stride_size = 1, window_length = 10, name = ''):

    if name:
        print('Generating strides for {}'.format(name))

    end = video_array.shape[-1] - window_length
    windows = []
    for i in range(0, end, stride_size):
        x = video_array[..., i:i+window_length]
        transformed_x = []
        for j in range(x.shape[-1]):
            temp = x[..., j]
            temp_transformed = transform(temp)
            transformed_x.append(temp_transformed)
        x = torch.cat(transformed_x)
        windows.append(torch.unsqueeze(x,0))
        # shape  = (10,227,227) 
    
    windows = torch.cat(windows)
    return windows
    
# @cache.memoize(typed=True, tag='loading')
def load_data_from_file(filename):

    videodata = sio.loadmat(str(filename))  
    videodata = videodata['vol']
    return videodata

class FileDataset(Dataset):

    def __init__(self, train_path, test_path, train = True):
        
        path_to_use = train_path if train else test_path

        self.filenames = glob.glob(str(path_to_use) + '/*')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, ndx):

        video_matrix = load_data_from_file(self.filenames[ndx])
        stride_1_set = generate_stride_set(video_matrix, stride_size=1, name=self.filenames[ndx])
        stride_2_set = generate_stride_set(video_matrix, stride_size=2, name=self.filenames[ndx])
        stride_3_set = generate_stride_set(video_matrix, stride_size=3, name=self.filenames[ndx])

        data_set = torch.unsqueeze(torch.cat([stride_1_set, stride_2_set, stride_3_set]),1)
        data_set = data_set[torch.randperm(data_set.size()[0])]
    

        return data_set
    

class AnomalyDataset(Dataset):
    
    def __init__(self, X, train=True, fraction = 0.8):

        pivot = int(fraction * len(X))
        if train:
            self.X = X[:pivot]
        else:
            self.X = X[pivot:]

    def __len__(self):

        return len(self.X)
    
    def __getitem__(self, ndx):
        return self.X[ndx, ...]


