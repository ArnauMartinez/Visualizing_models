
import matplotlib.pyplot as plt
import matplotlib_inline 
import tensorflow as tf
import tensorflow.keras as K
import os
import numpy as np
from tensorflow.keras.layers import Dense 
from dnn_app_utils_v3 import load_data, sigmoid
from tensorflow.python.ops.numpy_ops import np_config
from Model_architecture import *

np_config.enable_numpy_behavior()


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1)
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1)

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

model=CatModel([20,7,5,1])
model.build(train_x.shape)

opt=K.optimizers.SGD(learning_rate=0.0075)

loss_fn=K.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer=opt,loss=loss_fn,metrics=['accuracy'])

history=model.fit(train_x,train_y.T, batch_size=len(train_x),epochs=2500,validation_data=(test_x,test_y.T),validation_batch_size=len(test_x))
model.save_weights('Model_L-layers_weights.h5') 


plt.figure().clear()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model_accuracy')
plt.xlabel('iterations')
plt.ylabel('Accuracy')
plt.legend(['train','test'],loc='upper left')
#plt.show()
plt.savefig('Model_accuracy_L')


plt.figure().clear()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model_loss')
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.legend(['train','test'],loc='upper left')
#plt.show()
plt.savefig('Model_loss_L')
