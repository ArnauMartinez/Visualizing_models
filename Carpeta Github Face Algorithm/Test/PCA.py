import csv
import pandas as pd
from Model.FaceNet import FaceNet
import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from PIL import Image
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns



model=FaceNet(last_lyr=False,width_mult=0.75)
model.load_state_dict(torch.load('Fine_tuned_params_v2.pt',map_location='cpu'),strict=False)
model.eval()


csv_file=pd.read_csv('Dataset_dvd2.csv')

transform=transforms.Compose([ToTensor()])
length=max(csv_file['subject'])+1



embeddings=pd.read_csv('Embeddings.csv')

X=embeddings.iloc[:,:-1]
Y=embeddings.iloc[:,-1]

pca=PCA(n_components=2)
pca_result=pca.fit_transform(X.values)
pca_result.shape


plt.figure(figsize=(16,10))
sns.scatterplot(
    x=pca_result[:,0], y=pca_result[:,1],
    hue=Y,
    palette=sns.color_palette("hls", len(Y)),
    data=pca_result,
    legend="full",
    alpha=0.3
)
plt.show()
