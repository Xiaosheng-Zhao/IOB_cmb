import numpy as np
from torch.utils.data import Dataset
import torch

### Prepare the data
class CMBDataset(Dataset):
    """Custom Dataset for loading Ball images"""

    def __init__(
        self, 
        data_dir,
        data_mean_dir,
        para_dir, 
        err_dir,
        transform=None):

        self.data_dir = data_dir
        self.para_dir = para_dir
        self.data_mean_dir = data_mean_dir
        self.err_dir = err_dir
        self.transform = transform

        self.data = np.load(self.data_dir)[:,:]
        self.data_mean = np.load(self.data_mean_dir)[:,:]
        self.err = np.load(self.err_dir)[:,:]
        self.para = np.load(self.para_dir)[:,:]
        self.len = self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index,:]
        data = np.expand_dims(data,axis=0) # only one channel
        data_mean = self.data_mean[index,:]
        data_mean = np.expand_dims(data_mean,axis=0) 
        err = self.err[index,:]

        para = self.para[index]
        
        if self.transform is not None:
            data = self.transform(data)
            data_mean = self.transform(data_mean)
        
        return torch.tensor(data,dtype=torch.float32),torch.tensor(data_mean,dtype=torch.float32),torch.tensor(para,dtype=torch.float32),torch.tensor(err,dtype=torch.float32)

    def __len__(self):
        return self.len


