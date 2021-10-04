from torchvision import transforms
from torchvision.transforms.transforms import RandomCrop
from Model.FaceNet import FaceNet
import torch
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor, Resize
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import csv
import numpy as np


model=FaceNet(last_lyr=False,width_mult=0.75)
#model.load_state_dict(torch.load('Fine_tuned_params_256x171.pt',map_location='cpu'),strict=False)
model.load_state_dict(torch.load('Fine_tuned_params_256x171_v2.pt',map_location='cpu'),strict=False)
model.eval()



def create_matrix(persons_in_database,csv_file):
    

    transform=transforms.Compose([ToTensor(),Resize(size=(256,171))])
    matrix=torch.zeros(size=(persons_in_database,1280))
    indexer=csv_file.index
    for e in range(persons_in_database):
        output=torch.zeros(size=(1,1280))
        condition_1=csv_file['subject']==e 
        condition=condition_1
        indices=indexer[condition].to_list()
        for index in indices:
            img=Image.open(csv_file.iloc[index,0])
            img=transform(img)
            output+=model(img.unsqueeze(0))
        
        output=output/len(indices)
        print(e)
        matrix[e,:]=output 
    print('matrix_created')
    return matrix

def create_matrix_fa(persons_in_database,csv_file):
    transform=transform=transforms.Compose([ToTensor(),Resize(size=(256,171))])
    matrix=torch.zeros(size=(persons_in_database,1280))
    indexer=csv_file.index
    for e in range(persons_in_database):
        output=torch.zeros(size=(1,1280))
        condition_1=csv_file['subject']==e
        condition_2=csv_file['pose']=='fa'
        condition=condition_1 & condition_2
        indices=indexer[condition].to_list()
        for index in indices:
            img=Image.open(csv_file.iloc[index,0])
            img=transform(img)
            output+=model(img.unsqueeze(0))
        
        output=output/len(indices)
        print(e)
        matrix[e,:]=output
    return matrix


def distance(img,matrix,dist_num,model=model):
    pred=model(img)
    x=torch.cdist(pred,matrix)
    value,indices=torch.min(x,dim=1)
    if value>dist_num:
        indices='unknown'
    return value,indices

def evaluate(persons_in_database, valid_dl,csv_file,dist_num,matrix=None):
    
    if matrix==None:
        matrix=create_matrix(persons_in_database,csv_file)
    max_dist=0
    x=0
    false_negatives=0
    false_positives=0
    error=0
    cheers=0

    for i,(xb,yb) in enumerate(valid_dl):
        dist,indices=distance(xb,matrix,dist_num)
        if dist>max_dist:
            max_dist=dist
        if yb >(persons_in_database-1):
            yb='unknown'


        if indices==yb:
            x+=1
            if indices!='unknown':
                cheers+=1
        elif yb!='unknown':
            if indices=='unknown':
                false_negatives+=1
            else:
                error+=1    
        elif yb=='unknown':
            false_positives+=1

        print(i,x,error, false_positives, false_negatives,cheers)
        if i%1000==0:
            print(i,max_dist)
    return [i,x,error,false_positives,false_negatives,cheers]
 


           




def evaluate_params(params_doc_list,model,persons_in_database,csv_file,dist_num,iterations=10):
    header=['params_doc','persons_in_db','distance_threshold','images_evaluated','total_success','errors_both_db','false_positives','false_negatives','pairing_success']
    with open('Test_results.csv','w')as f:
        writer=csv.writer(f)
        writer.writerow(header)

        for doc in params_doc_list:
            row=[]
            model.load_state_dict(torch.load(str(doc),map_location='cpu'),strict=False)
            model.eval()
            iterations_value=np.zeros(shape=(iterations, len(header)-3 ))
            for iteration in range(iterations):
                iterations_value[iteration,:]=evaluate(persons_in_database,valid_dl,csv_file,dist_num)
                mean=np.mean(iterations_value,axis=0)
            row=[doc,persons_in_database,dist_num]+list(mean)
            writer.writerow(row)
            print(1)
    
            
        

params_doc_list=['Fine_tuned_params_256x171.pt','Model_state_dict_256x171.pt','Model_state_dict_256x171_v2.pt']
persons_in_database=50
csv_file=pd.read_csv('Dataset_dvd2.csv')  
transform=transforms.Compose([ToTensor(),Resize(size=(256,171)),])#RandomCrop(size=(256,171),padding=(40,26),padding_mode='reflect')])
dataset=ImageFolder('colorferet/dvd2/data/images',transform=transform)
valid_dl=DataLoader(dataset,batch_size=1,shuffle=True) 
dist_num=19
evaluate_params(params_doc_list,model,persons_in_database,csv_file,dist_num,iterations=1)

matrix=create_matrix_fa(persons_in_database,csv_file)
evaluate(persons_in_database,valid_dl,csv_file,dist_num,matrix)



















matrix=create_matrix(268,csv_file)
triplets_dict={}
a=torch.cdist(matrix,matrix)
for e in range(len(a)):
    a[e,e]=1000



values, indices=torch.min(a,dim=1)
for i,e in enumerate(indices):
    triplets_dict[str(i)]=e


for a,b in triplets_dict.items():
    print(a,b)


triplets_dict['202']

triplets_dict['239']

triplets_dict['130']

values.sort()

triplets_dict['62']