import os
from numpy.core.fromnumeric import std
import torch
from torch.utils.data import DataLoader, Dataset, dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import ImageFile, Image
import random

def get_train_dl(datadir,batch_size):
    train_dataset=ImageFolder(datadir, transform=ToTensor())
    train_dl=DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
    return train_dl






def calculate_mean_std(datadir, normalize=True, train_dl=None):

    if train_dl is None:
        dataset=ImageFolder(datadir, transform=ToTensor())
        train_dl=DataLoader(dataset,batch_size=55)
    means=torch.zeros((3,len(train_dl)))
    stds=torch.detach(means).clone()
    
    for i,(xb,__) in enumerate(train_dl):
        mean=xb.mean(3).mean(2).mean(0)
        standard=std(xb.numpy(),axis=(0,2,3))
        means[:,i]=mean
        stds[:,i]=torch.from_numpy(standard)
        print('Batch [{}/{}]'.format(i+1,len(train_dl)))
            
    mean=means.mean(1)
    stds=stds.mean(1)
    if normalize:
        dataset=ImageFolder(datadir,transform=transforms.Compose([ToTensor(),Normalize(mean,stds)]))
        train_dl=DataLoader(dataset,batch_size=55,shuffle=True)
        torch.save([mean,stds], 'Preprocess_data.txt')
        return train_dl
    else:
        return (mean,stds)

def normalize_data(datadir, File,Dataloader=True):
    data=torch.load(File)
    dataset=ImageFolder(datadir, transform=transforms.Compose([ToTensor(),Normalize(*tuple(data))]) )
    train_dl=DataLoader(dataset, batch_size=55, shuffle=True ) if Dataloader else dataset
    return train_dl
        





        










#x=[0,0,0]
#with open('Dataset_info.txt','r') as f:
    #c=f.readlines()
    #for i,e in enumerate(c):
        #e=e.split(' ')
        
        #idx=e.index('average:')
        #x[i]=float(e[idx+1])
    #x=torch.tensor(x)









