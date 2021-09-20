
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D


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


def ConvModel(input_shape=(64,64,3)):
    
    layers_list=get_layers_info()
    input=K.Input(shape=input_shape)
    x=input
    for e in layers_list:
           x=e(x)
    
    model=K.Model(inputs=input,outputs=x)
    print(K.utils.plot_model(model, show_shapes=True))
    return model

      
def get_layers_info():
    layers_name={
            'Conv2D': Conv2d,
            'MaxPooling': MaxPool2d,
            'FC': fullyconected        
    }

    layers_list=[]
    e_1=0


    x=int(input('How many layers would you like to be on your model? (integer): \n'))
    layer_type=[input('Which type of layer do you want your [{}/{}] layer to be?\n{}'.format(e+1,x,layers_name.keys()))for e in range(x)]
    print('______________________________________________________________')

    for i,e in enumerate(layer_type):
        print('{} layer configuration\n'.format(e))
        
        if e_1=='Conv2D' and e=='FC':
            layers_list.append(K.layers.Flatten())
            e_1=0

        layers_list.append(layers_name.get(e)())

        if i!=(x-1):
            layers_list.append(layers.ReLU())
        else:
            layers_list.append(layers.Activation('sigmoid'))

        if e=='Conv2D':
            e_1=e
        
        print('________________________________________________')
    return layers_list




def Conv2d():
    
    filters=int(input('How many output filters do you want for this layer?: \n'))
    kernel_size=int(input('kernel size: \n'))
    stride= int(input('Stride: \n'))
    padding=input("padding: (valid/same) \n")
    return Conv2D(filters,kernel_size,stride,padding)


def MaxPool2d():
    
    pool_size=int(input('Pool size: \n'))
    stride=pool_size
    return MaxPool2D(pool_size,stride)

def fullyconected():

    output_dim=int(input('Number of output neurons: \n'))
    return Dense(output_dim)    

