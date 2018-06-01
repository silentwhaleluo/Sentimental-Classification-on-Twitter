
# coding: utf-8

#import packages and set parameters.
from keras import models
from keras import layers
from keras.layers import Dense, Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping


from keras import regularizers
#configure GPU，if you do not use Tensorflow-GPU delete this part
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config)) 
print('configed')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path='\SOT\'

ori_df=pd.read_csv(path+"BIA652_train_1000_remove.csv")
ori_df.head()

print(ori_df.shape)
y=np.array(ori_df['Sentiment'])
data=np.array(ori_df)
x=data[:,2:]
print(x.shape)
print(y.shape)

y_train=y[:800]
x_train=x[:800,:]
y_test=y[800:]
x_test=x[800:,:]
print(x_train.shape)

input_shape = x_train[1,].shape
batch_size = 128
epochs = 30
validation_split = 0.2
EarlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='min')

#build model, train, and save the weight
#del model
model = models.Sequential()
model.add(layers.Dense(1024,activation='relu', input_shape=input_shape))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128,activation='relu',
                       kernel_regularizer = regularizers.l2(0.04),bias_regularizer = regularizers.l2(0.06)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid',
                       kernel_regularizer = regularizers.l2(0.04),bias_regularizer  = regularizers.l2(0.06)))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history_ori = model.fit(x_train,y_train,epochs = epochs,
                   batch_size = batch_size, validation_split = validation_split, callbacks=[EarlyStopping])

score_ori = model.evaluate(x_test, y_test, batch_size=batch_size)

print(score_ori)
model.save(path+'results\\'+'Step4_NN model_original dataset.h5')
print('Model Saved to',path+'results\\'+)

# create loss plots
plt.figure(figsize=(12,10)) 
epochs = range(1, len(history_ori.history['acc']) + 1)
plt.plot(epochs, history_ori.history['loss'], label='Training Loss') 
plt.plot(epochs, history_ori.history['val_loss'], label='Validation Loss') 

plt.title('Loss of Neural Network(Orginal Dataset with 4583 variables)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(path+'results\\'+'Step4_nn_original_loss.png')
plt.show()
print('Plot saved')

# create accuracy plots
plt.figure(figsize=(12,10)) 
epochs = range(1, len(history_ori.history['acc']) + 1)
plt.plot(epochs, history_ori.history['acc'], label='Training Accuracy') 
plt.plot(epochs, history_ori.history['val_acc'], label='Validation Accuracy') 
plt.title('Accuracy of Neural Network(Orginal Dataset with 4583 variables)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(path+'results\\'+'Step4_nn_original_accuracy.png')
plt.show()
print('Plot saved')

df_DR=pd.read_csv(path+'BIA652_train_1000_DR.csv')
print(df_DR.head())
np_dr=np.array(df_DR)
x=np_dr[:,1:-1]
y=np_dr[:,1]
y_train=y[:800]
x_train=x[:800,:]
y_test=y[800:]
x_test=x[800:,:]
print(x_train.shape)
print(y.sum())

input_shape = x_train[1,].shape
batch_size = 128
epochs = 30
validation_split = 0.2

from keras.callbacks import EarlyStopping
EarlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='min')

from keras import regularizers

del model
model = models.Sequential()
model.add(layers.Dense(256,activation='relu', input_shape=input_shape,))
model.add(layers.Dense(128,activation='relu',))
model.add(layers.Dense(64,activation='relu',))
model.add(layers.Dense(32,activation='relu',))
model.add(layers.Dense(16,activation='relu',))
model.add(layers.Dense(1,activation='sigmoid',))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'],
             )

history_pca = model.fit(x_train,y_train,epochs=epochs,
                       batch_size = batch_size,
                       validation_split = validation_split)

score_pca = model.evaluate(x_test, y_test, batch_size=128)

print(score_pca)

model.save(path+'results\\'+'Step4_NN model_pca dataset.h5')
print('Model Saved to',path+'results\\')

# create loss plots
import matplotlib.pyplot as plt
plt.figure(figsize=(12,10)) 
epochs = range(1, len(history_pca.history['acc']) + 1)
plt.plot(epochs, history_pca.history['loss'], label='Training Loss') 
plt.plot(epochs, history_pca.history['val_loss'], label='Validation Loss') 

plt.title('Loss of Neural Network(PCA Dataset with 369 variables')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(path+'results\\'+'Step4_nn_PCA_loss.png')
plt.show()
print('Plot saved')

# create accuracy plots
plt.figure(figsize=(12,10)) 
epochs = range(1, len(history_pca.history['acc']) + 1)
plt.plot(epochs, history_pca.history['acc'], label='Training Accuracy') 
plt.plot(epochs, history_pca.history['val_acc'], label='Validation Accuracy') 
plt.title('Accuracy of Neural Network(PCA Dataset with 369 variables')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(path+'results\\'+'Step4_nn_PCA_accuracy.png')
plt.show()
print('Plot saved')
