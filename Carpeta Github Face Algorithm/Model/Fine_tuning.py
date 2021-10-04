from Model.FaceNet import FaceNet
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from Model.Preprocessing import normalize_data
import time
from Model.Dataset import Feret_database_triplets
import matplotlib.pyplot as plt
import torch.nn.functional as F

csv_file='Dataset_k_1.csv
model=FaceNet(last_lyr=False)
model.load_state_dict(torch.load('./Model/model_params_definitive.pt'),strict=False)




dataset=Feret_database_triplets(csv_file)

train_dl=DataLoader(dataset, batch_size=16,shuffle=True)


def loss_batch_triplet(model,anchor_imgs,Pos_imgs,Neg_imgs,loss_fn,opt=None, metric=None):
    A=model(anchor_imgs)
    P=model(Pos_imgs)
    N=model(Neg_imgs)

    loss=loss_fn(A,P,N,1)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    

    metric_result=None
    if metric is not None:
        metric_result=metric(A,P,N)
    
    return (A,P,N), loss, metric_result


    

def triplet_convergence(model,train_dl,loss_fn, epochs,lr=0.001,opt=None, metric=None):
    

    losses=torch.zeros((epochs,len(train_dl)),dtype=float)

    if opt is not None:
        opt=opt(model.parameters(),lr)
        model.train()
    else:
        model.eval()

    metrics=None
    if metric is not None:
        metrics=torch.zeros((epochs, len(train_dl)),dtype=float)

    for e in range(epochs):
        dataset=Feret_database_triplets(csv_file)


        train_dl=DataLoader(dataset, batch_size=16,shuffle=True)
        for i,data in enumerate(train_dl):
            anchor_imgs,Pos_imgs,Neg_imgs=data
            preds,loss,metric_result=loss_batch_triplet(model,anchor_imgs,Pos_imgs,Neg_imgs,loss_fn,opt=opt,metric=metric)
            losses[e,i]=loss

            if metric is not None:
                metrics[e,i]=metric_result
            
            print('Cost in epoch {} and batch {} is: {}'.format(e+1,i+1,loss))
            if metric is not None:
               print(f'Accuracy is: {metric_result}')
            print('______________________________')
    
    return losses,metrics


loss_fn=F.triplet_margin_loss
optimizer=torch.optim.Adam
losses,metrics=triplet_convergence(model,train_dl,loss_fn,10,opt=optimizer)


torch.save(model.state_dict(),'Face_params.pt')
