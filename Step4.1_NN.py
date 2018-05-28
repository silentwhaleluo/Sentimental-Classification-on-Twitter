import pandas as pd
import numpy as np
path='\\SOT\\'

#Firstly, using original dataset to train the neural network.
ori_df=pd.read_csv(path+"train_1000.csv")
ori_df.head()

#test the dataset
ori_df['Sentiment'].value_counts()
ori_df.keys()


# In[7]:


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


# In[38]:


#import packages and set parameters.
from keras import models
from keras import layers
from keras.layers import Dense, Dropout
from keras import regularizers
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config)) 
print('configed')
input_shape = x_train[1,].shape
batch_size = 128
epochs = 30


# In[40]:


#build model, train, and save the weight
del model
model = models.Sequential()
model.add(layers.Dense(1024,activation='relu', input_shape=input_shape))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128,activation='relu',))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid',))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history_ori = model.fit(x_train,y_train,epochs = epochs,
                   batch_size = batch_size)

score_ori = model.evaluate(x_test, y_test, batch_size=batch_size)

print(score_ori)
model.save(path+'Step4_NN model_original dataset.h5')
print('Model Saved to',path)


# In[41]:


# create plots
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6)) 
epochs = range(1, len(history_ori.history['acc']) + 1)
plt.plot(epochs, history_ori.history['loss'], label='Training Loss') 
plt.plot(epochs, history_ori.history['acc'], label='Training Accuracy') 

plt.title('Traning of Neural Newwork with original dataset( 4583 dimensions )')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(path+'Step4_history_original dataset.png')
plt.show()
print('Plot saved')


# Secondly, using pca results, compare the training results.
pca_df=pd.read_csv(path+"BIA652_train_1000_DR.csv")
pca_df.head()

#test the dataset
print(pca_df['sentiment'].value_counts())
pca_df.keys()

print(pca_df.shape)
y=np.array(pca_df['sentiment'])
data=np.array(pca_df)
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


# In[51]:


#build model, train, and save the weight
#del model
model = models.Sequential()
model.add(layers.Dense(256,activation='relu', input_shape=input_shape))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128,activation='relu',))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid',))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history_pca = model.fit(x_train,y_train,epochs = epochs,
                   batch_size = batch_size)

score_pca = model.evaluate(x_test, y_test, batch_size=batch_size)

print(score_pca)
model.save_weights(path+'Step4_NN model_pca dataset.h5')
print('Model Saved to',path)

# create plots
plt.figure(figsize=(8,6)) 
epochs = range(1, len(history_pca.history['acc']) + 1)
plt.plot(epochs, history_pca.history['loss'], label='Training Loss') 
plt.plot(epochs, history_pca.history['acc'], label='Training Accuracy') 

plt.title('Traning of Neural Newwork with pca dataset( 369 dimensions )')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(path+'Step4_history_pca dataset.png')
plt.show()
print('Plot saved')
