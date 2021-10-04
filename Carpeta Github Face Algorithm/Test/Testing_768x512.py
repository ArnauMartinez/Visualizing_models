from torch._C import Value
from torchvision import transforms
from torchvision.transforms.transforms import RandomCrop
from Model.FaceNet import FaceNet
import torch
import pandas as pd
import random
from PIL import Image
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
csv_file=pd.read_csv('Dataset_dvd2.csv')

model=FaceNet(last_lyr=False,width_mult=0.75)
model.load_state_dict(torch.load('Fine_tuned_params_v2.pt',map_location='cpu'),strict=False)
model.eval()




transform=transforms.Compose([ToTensor()])
length=max(csv_file['subject'])+1
matrix=torch.zeros(size=(length,1280))
for e in range(length):
    indexer=csv_file.index
    condition_1=csv_file['subject']==e 
    condition_2= csv_file['pose']=='fa'
    condition=condition_1 & condition_2
    indices=indexer[condition].to_list()
    index=indices[0]
    img=Image.open(csv_file.iloc[index,0])
    img=transform(img)
    output=model(img.unsqueeze(0))
    matrix[e,:]=output 
    print(e)

    
transform=transforms.Compose([ToTensor(),RandomCrop(size=(768,512),padding=(120,80),padding_mode='reflect')])
dataset=ImageFolder('colorferet/dvd2/data/images',transform=transform)
valid_dl=DataLoader(dataset,batch_size=1,shuffle=True)


def cosine_distance(img,matrix,model=model):
    pred=model(img)
    x=F.cosine_similarity(pred,matrix,dim=1).unsqueeze(0)
    value,indices=torch.max(x,dim=1)
    if value<0.9:
        indices='unknown'
    return (value,indices)
    

def distance(img,matrix,model=model):
    pred=model(img)
    x=torch.cdist(pred,matrix)
    value,indices=torch.min(x,dim=1)
    return value,indices
max_dist=0
x=0
for i,(xb,yb) in enumerate(valid_dl):
    dist,indices=distance(xb,matrix)
    if dist>max_dist:
        max_dist=dist
    x+=indices==yb
    print(i,x)
    if i==999:
        print(max_dist)
        break
    



a=torch.cdist(matrix,matrix)
for e in range(len(a)):
    a[e,e]=1000
    

values, indices=torch.min(a,dim=1)


dist_dict={}

for e in range(len(indices)):
    dist_dict[str(e)]=indices[e]

dist_dict['202']

