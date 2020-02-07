# -*- coding: utf-8 -*-
"""
Created on Mon May  6 18:18:41 2019

@author: WTDSP_60000NTD
"""
import numpy as np 
import pandas as pd
from keras import initializers
from keras import optimizers
from keras.models import Sequential  
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.utils import np_utils 
from matplotlib import pyplot as pl
import scipy.io
from keras.utils import np_utils
from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('train.mat')
x_train=mat['x_train']
y_train=mat['y_train']


      #Train 預處理 第一個數字
x_train_first=x_train[:,0:48,0:42]
x_resize_first=np.zeros(((x_train_first.shape[0], 28, 28)))

for i in range(x_train_first.shape[0]):
    x_process_first=x_train_first[i]
    x_resize_first[i]=scipy.misc.imresize(x_process_first,size=(28,28))

x_train_go_first=x_resize_first.reshape(5838,28,28,1).astype('float32')
      
      #Train 預處理 第二個數字
x_train_second=x_train[:,0:48,44:68]
x_resize_second=np.zeros(((x_train_second.shape[0], 28, 28)))

for i in range(x_train_second.shape[0]):
    x_process_second=x_train_second[i]
    x_resize_second[i]=scipy.misc.imresize(x_process_second,size=(28,28))

x_train_go_second=x_resize_second.reshape(5838,28,28,1).astype('float32')
      
      #Train預處理 第三個數字
x_train_third=x_train[:,0:48,68:96]
x_resize_third=np.zeros(((x_train_third.shape[0], 28, 28)))

for i in range(x_train_third.shape[0]):
    x_process_third=x_train_third[i]
    x_resize_third[i]=scipy.misc.imresize(x_process_third,size=(28,28))

x_train_go_third=x_resize_third.reshape(5838,28,28,1).astype('float32')
      
      #Train預處理 第四個數字
x_train_fourth=x_train[:,0:48,96:136]
x_resize_fourth=np.zeros(((x_train_fourth.shape[0], 28, 28)))

for i in range(x_train_fourth.shape[0]):
    x_process_fourth=x_train_fourth[i]
    x_resize_fourth[i]=scipy.misc.imresize(x_process_fourth,size=(28,28))

x_train_go_fourth=x_resize_fourth.reshape(5838,28,28,1).astype('float32')
      #Train預處理 四個數字合併
x_train_go=np.concatenate((x_train_go_first, x_train_go_second, x_train_go_third, x_train_go_fourth), axis = 0)

      #Train預處理 label
y_train_first=y_train[:,0:1]
y_train_second=y_train[:,1:2]
y_train_third=y_train[:,2:3]
y_train_fourth=y_train[:,3:4]
y_train_con=np.concatenate((y_train_first, y_train_second, y_train_third, y_train_fourth), axis = 0)
y_train_onehot = np_utils.to_categorical(y_train_con,19)


    #Test 預處理 第一個數字
x_test_first=x_train[4838+500:5838,0:48,0:42]
x_test_resize_first=np.zeros(((x_test_first.shape[0], 28, 28)))

for i in range(x_test_first.shape[0]):
    x_test_process_first=x_test_first[i]
    x_test_resize_first[i]=scipy.misc.imresize(x_test_process_first,size=(28,28))

x_test_go_first=x_test_resize_first.reshape(1000-500,28,28,1).astype('float32')
    #Test 預處理第二個數字
x_test_second=x_train[4838+500:5838,0:48,44:68]
x_test_resize_second=np.zeros(((x_test_second.shape[0], 28, 28)))

for i in range(x_test_second.shape[0]):
    x_test_process_second=x_test_second[i]
    x_test_resize_second[i]=scipy.misc.imresize(x_test_process_second,size=(28,28))

x_test_go_second=x_test_resize_second.reshape(1000-500,28,28,1).astype('float32')
    #Test 預處理第三個數字
x_test_third=x_train[4838+500:5838,0:48,68:96]
x_test_resize_third=np.zeros(((x_test_third.shape[0], 28, 28)))

for i in range(x_test_third.shape[0]):
    x_test_process_third=x_test_third[i]
    x_test_resize_third[i]=scipy.misc.imresize(x_test_process_third,size=(28,28))

x_test_go_third=x_test_resize_third.reshape(1000-500,28,28,1).astype('float32')
    #Test 預處理地四個數字
x_test_fourth=x_train[4838+500:5838,0:48,96:136]
x_test_resize_fourth=np.zeros(((x_test_fourth.shape[0], 28, 28)))

for i in range(x_test_fourth.shape[0]):
    x_test_process_fourth=x_test_fourth[i]
    x_test_resize_fourth[i]=scipy.misc.imresize(x_test_process_fourth,size=(28,28))

x_test_go_fourth=x_test_resize_fourth.reshape(1000-500,28,28,1).astype('float32')

    #Test預處理 四個數字合併
x_test_go=np.concatenate((x_test_go_first, x_test_go_second, x_test_go_third, x_test_go_fourth), axis = 0)

    #Test預處理 label 
y_test_first=y_train[4838+500:5838,0:1]
y_test_second=y_train[4838+500:5838,1:2]
y_test_third=y_train[4838+500:5838,2:3]
y_test_fourth=y_train[4838+500:5838,3:4]
y_test_con=np.concatenate((y_test_first, y_test_second, y_test_third, y_test_fourth), axis = 0)
y_test_onehot = np_utils.to_categorical(y_test_con,19)



    #訓練模型
model = Sequential()
model.add(Conv2D(64, kernel_size=(7, 7), padding='same', input_shape=(28,28,1), kernel_initializer=initializers.he_normal(seed=0)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (5, 5), padding='same', kernel_initializer=initializers.he_normal(seed=0)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=0)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(128, kernel_initializer=initializers.he_normal(seed=0), activation='relu'))
model.add(Dropout(0.14))
model.add(Dense(64, kernel_initializer=initializers.he_normal(seed=0), activation='relu'))
model.add(Dropout(0.14))
model.add(Dense(32, kernel_initializer=initializers.he_normal(seed=0), activation='relu'))
model.add(Dropout(0.14))
model.add(Dense(19,activation='softmax', kernel_initializer=initializers.he_normal(seed=0)))

     #訓練優化
sgd = optimizers.SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
"""optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)"""
model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
model.fit(x_train_go, y_train_onehot, batch_size=32, epochs=70, verbose=1)

