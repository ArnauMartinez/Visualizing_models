
import torch.nn.functional as F
from Model.FaceNet import FaceNet
import torch
from Model.Dataset import *
import numpy as np
import matplotlib.pyplot as plt
from Model.Preprocessing import *
import torch.nn as nn

def create_model():
    model=FaceNet()
    return model


optimizer= torch.optim.Adam
datadir='colorferet/dvd1/data/images'

train_dl=normalize_data(datadir,'Preprocess_data_dvd1.txt')
loss_fn=F.cross_entropy
learning_rate=0.0005

def accuracy(outputs,labels):
    __,preds=torch.max(outputs,dim=1)
    return torch.sum(preds==labels).item()/len(labels)



def loss_batch(model, xb,yb, loss_fn,  metric=None, opt=None):
    preds=model(xb)
    loss=loss_fn(preds,yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    
    metric_result=None
    if metric is not None:
        metric_result=metric(preds,yb)
    
    return preds, float(loss), float(metric_result)


def fit(model, train_dl, loss_fn, lr=0.001, epochs=1, optimizer=None, metric=None):
    
    if optimizer is not None:
        model.train()
        opt=optimizer(model.parameters(),lr,weight_decay=0.000005)
    else:
        model.eval()
        opt=optimizer
    losses=np.zeros(((len(train_dl),epochs)), dtype=float)

    if metric is not None:
        metric_eval=np.zeros(((len(train_dl),epochs)), dtype=float)
        metric_Avg=0
    
    for i in range(epochs):

        for e,(xb,yb) in enumerate(train_dl):
            
            __,loss,metric_result=loss_batch(model,xb,yb,loss_fn,metric,opt)
            losses[e,i]=loss
            if metric is not None:
                metric_eval[e,i]=metric_result
                metric_Avg=0.1*metric_result+0.9*metric_Avg

            #if e % 25 ==0:
            print('Cost in epoch {} and batch {} is: {}'.format(i+1,e+1,loss))
            print(f'Accuracy is: {metric_Avg}')
            print('______________________________')

    return metric_eval,losses
    


    
model=create_model()
model.load_state_dict(torch.load('Model/model_params.pt'))

metric_eval,losses= fit(model,train_dl,loss_fn,epochs= 1,optimizer=optimizer,metric=accuracy)
torch.save(model.state_dict(),'./Model/model_params_definitive.pt')


evaluation,losses_eval=fit(model,train_dl,loss_fn, epochs=1,metric=accuracy)

evaluation.mean()










