
import matplotlib.pyplot as plt
import matplotlib_inline 
import tensorflow as tf
import tensorflow.keras as K
import os
import numpy as np
from tensorflow.keras.layers import Dense 
from dnn_app_utils_v3 import load_data, sigmoid
from tensorflow.python.ops.numpy_ops import np_config
from Model_architecture import CatModel

np_config.enable_numpy_behavior()





train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

train_x_orig=train_x_orig/255
test_x_orig=test_x_orig/255


X_train=tf.reshape(train_x_orig,[train_x_orig.shape[0],-1])
X_test=tf.reshape(test_x_orig,[test_x_orig.shape[0],-1])

model=CatModel([7,1])
model.build(X_train.shape)

opt=K.optimizers.SGD(learning_rate=0.002)

loss_fn=K.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer=opt,loss=loss_fn,metrics=['accuracy'])

history=model.fit(X_train,train_y.T, batch_size=len(X_train),epochs=2500,validation_data=(X_test,test_y.T),validation_batch_size=len(X_test))
model.save_weights('Model_2layers_weights.h5')


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model_accuracy')
plt.xlabel('iterations')
plt.ylabel('Accuracy')
plt.legend(['train','test'],loc='upper left')
plt.savefig('Model_accuracy')


plt.figure().clear()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model_loss')
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.legend(['train','test'],loc='upper left')
plt.savefig('Model_loss')

