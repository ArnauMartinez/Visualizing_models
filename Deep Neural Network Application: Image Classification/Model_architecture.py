
import tensorflow as tf
import numpy as np
import tensorflow.keras as K
from tensorflow.keras import losses
from tensorflow.keras.layers import Dense 
from tensorflow.keras import regularizers


class CatModel(K.Model):
    def __init__(self,layers_size,inp_size=12288,activation=None):
        super(CatModel,self).__init__()
        self.model=K.Sequential()
        self.model.add(K.Input(shape=(inp_size,)))
        for data in layers_size:
            if data != layers_size[-1]:
                layer=Dense(data,activation='relu')
            else:
                layer=Dense(data,activation=activation)
            self.model.add(layer)
        
        
    
    def call(self, inputs):
        x=self.model(inputs)
        return x


    

