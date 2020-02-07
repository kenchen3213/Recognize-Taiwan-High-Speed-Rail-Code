# -*- coding: utf-8 -*-
"""
Created on Thu May  9 19:26:11 2019

@author: Abby
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
from keras.models import load_model

model = load_model('test_model.h5')
print(model.summary())
mat = scipy.io.loadmat('test.mat')
x_test=mat['x_test']
y_test=mat['y_test']
     #Test 預處理 第一個數字
x_test_first=x_test[:,0:48,0:42]
x_test_resize_first=np.zeros(((x_test_first.shape[0], 28, 28)))

for i in range(x_test_first.shape[0]):
    x_test_process_first=x_test_first[i]
    x_test_resize_first[i]=scipy.misc.imresize(x_test_process_first,size=(28,28))

x_test_go_first=x_test_resize_first.reshape(2000,28,28,1).astype('float32')
    #Test 預處理第二個數字
x_test_second=x_test[:,0:48,44:68]
x_test_resize_second=np.zeros(((x_test_second.shape[0], 28, 28)))

for i in range(x_test_second.shape[0]):
    x_test_process_second=x_test_second[i]
    x_test_resize_second[i]=scipy.misc.imresize(x_test_process_second,size=(28,28))

x_test_go_second=x_test_resize_second.reshape(2000,28,28,1).astype('float32')
    #Test 預處理第三個數字
x_test_third=x_test[:,0:48,68:96]
x_test_resize_third=np.zeros(((x_test_third.shape[0], 28, 28)))

for i in range(x_test_third.shape[0]):
    x_test_process_third=x_test_third[i]
    x_test_resize_third[i]=scipy.misc.imresize(x_test_process_third,size=(28,28))

x_test_go_third=x_test_resize_third.reshape(2000,28,28,1).astype('float32')
    #Test 預處理地四個數字
x_test_fourth=x_test[:,0:48,96:136]
x_test_resize_fourth=np.zeros(((x_test_fourth.shape[0], 28, 28)))

for i in range(x_test_fourth.shape[0]):
    x_test_process_fourth=x_test_fourth[i]
    x_test_resize_fourth[i]=scipy.misc.imresize(x_test_process_fourth,size=(28,28))

x_test_go_fourth=x_test_resize_fourth.reshape(2000,28,28,1).astype('float32')

    #Test預處理 四個數字合併
x_test_go=np.concatenate((x_test_go_first, x_test_go_second, x_test_go_third, x_test_go_fourth), axis = 0)

    #Test預處理 label 
y_test_first=y_test[:,0:1]
y_test_second=y_test[:,1:2]
y_test_third=y_test[:,2:3]
y_test_fourth=y_test[:,3:4]
y_test_con=np.concatenate((y_test_first, y_test_second, y_test_third, y_test_fourth), axis = 0)
y_test_onehot = np_utils.to_categorical(y_test_con,19)


def count_wrong(x, y):
    
    count=0
    
    for i in range(len(x)):
        wrong=int(x[i]!=y[i])
        count+=wrong
   
    return count
    
def accuracy(x, y):
    
    wrong=0
    for i in range(x.shape[0]):
        
        for j in range(x.shape[1]):
            
            
            if x[i][j]!=y[i][j]:
                cc=1
                wrong+=cc
                break
    
    acc=1-(wrong/y.shape[0])        
    
    return acc
         
score=model.evaluate(x_test_go, y_test_onehot)
prediction=model.predict_classes(x_test_go)
prediction=prediction.reshape(-1,1)

print("2000 _pictures_total_wrong=", count_wrong(prediction, y_test_con))
print("one word accuracy :", score[1])

prediction_first=prediction[0:2000,:]
label_test_first=y_test_con[0:500,:]

prediction_second=prediction[2000:4000,:]
label_test_second=y_test_con[500:1000,:]

prediction_third=prediction[4000:6000,:]
label_test_third=y_test_con[1000:1500,:]

prediction_fourth=prediction[6000:8000,:]
label_test_fourth=y_test_con[1500:2000,:]

prediction_con=np.concatenate((prediction_first, prediction_second, prediction_third, prediction_fourth), axis = 1)
y_label=y_test


print("accuracy :", accuracy(prediction_con, y_label))

prediction_con_40=prediction_con[0:40,:]
y_label_40=y_label[0:40,:]

chars=['2','3','4','5','7','9','A','C','F','H','K','M','N','P','Q','R','T','Y','Z']
ix_to_chars={i:ch for i, ch in enumerate(chars)}

for i in range(0,40):
    
    a=[ix_to_chars[j] for j in prediction_con_40[i]] 
    b=[ix_to_chars[j] for j in y_label_40[i]] 
    
    print("predict=", a)
    print("answer=", b ,",wrong words=", count_wrong(prediction_con_40[i], y_label_40[i]))
    






















