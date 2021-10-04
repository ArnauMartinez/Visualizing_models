from Model.FaceNet import FaceNet
import torch
from Model.Preprocessing import normalize_data
from Model.Training import accuracy
import time
import matplotlib
model2=FaceNet()


model2.load_state_dict(torch.load('./Model/model_params.pt'))


datadir='./colorferet/dvd1/data/images'
file='Preprocess_data_dvd1.txt'
def evaluate(model, metric,datadir,file):
    dataset=normalize_data(datadir,file)
    metric_result=[]
    model.eval()
    for i,(xb, yb) in enumerate(dataset):

        preds=model(xb)
        
        metric_result.append(metric(preds,yb))
        print(i+1)
    return metric_result

evaluate(model2,accuracy, datadir,file)

def accuracy(outputs,labels):
    __,preds=torch.max(outputs,dim=1)
    return torch.sum(preds==labels).item()/len(labels)


model2=FaceNet()
model2.load_state_dict(torch.load('Model/model_params_definitive.pt'))


train_dl=normalize_data(datadir,file,False)
model2.eval()
times=[]


        



for xb,yb in train_dl:
    _,indices=torch.max(model2(xb.unsqueeze(0)),dim=1)
    indices==yb
    
    


model=FaceNet()

model.load_state_dict(torch.load('./Model/model_params.pt')) 



dataset=normalize_data(datadir,file)
metric_result=[]
model.eval()
for i,(xb, yb) in enumerate(dataset):

    preds=model(xb)
    
    metric_result.append(accuracy(preds,yb))
    print(metric_result)
    break


