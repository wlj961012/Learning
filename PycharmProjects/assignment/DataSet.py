from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image

class dataset(Dataset):

    def __init__(self,root_dir,label_file,transform=None):
        self.root_dir = root_dir
        self.label = np.loadtxt(label_file)
        #print(self.label.shape)
        self.transform = transform

    def __getitem__(self,index):
        img_path=os.path.join(self.root_dir,'%s.jpg'%index)
        img=Image.open(img_path)
        labels=self.label[index,:]
        #print(labels)
        if self.transform:
            img=self.transform(img)
        return img,labels


    def __len__(self):
        return self.label.shape[0]




