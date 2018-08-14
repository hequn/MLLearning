from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout, Multiply, Embedding, Lambda
from keras.layers import Conv2D, MaxPooling2D,PReLU
from keras import backend as K
import tensorflow as tf
import numpy as np
import sys
from keras.callbacks import *
import TYY_callbacks
from keras.optimizers import SGD, Adam


batch_size = 128
num_classes = 10
epochs = 50

isCenterloss = True
# isCenterloss = False



# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Maintain single value ground truth labels for center loss inputs
# Because Embedding layer only accept index as inputs instead of one-hot vector
y_train_value = y_train
y_test_value = y_test

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

inputs = Input(shape=(28,28,1))
x = Conv2D(32, (3,3))(inputs)
x = PReLU()(x)
x = Conv2D(64, (3,3))(x)
x = PReLU()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128)(x)
x = Dropout(0.5)(x)
x = Dense(2)(x)
ip1 = PReLU(name='ip1')(x)
ip2 = Dense(num_classes, activation='softmax')(ip1)
# inputs = Input(shape=(28,28,1))
# x = Conv2D(32, (3,3))(inputs)
# x = PReLU()(x)
# x = Conv2D(32, (3,3))(x)
# x = PReLU()(x)
# x = Conv2D(64, (3,3))(x)
# x = PReLU()(x)
# x = Conv2D(64, (5,5))(x)
# x = PReLU()(x)
# x = Conv2D(128, (5,5))(x)
# x = PReLU()(x)
# x = Conv2D(128, (5,5))(x)
# x = PReLU()(x)
# x = Flatten()(x)
# x = Dense(2)(x)
# ip1 = PReLU(name='ip1')(x)
# ip2 = Dense(num_classes, activation='softmax')(ip1)

model = Model(inputs=inputs, outputs=[ip2])
model.compile(loss="categorical_crossentropy",
              optimizer=SGD(lr=0.05),
              metrics=['mae','accuracy'])

def l2_loss_fun(x):
    print(x)
    ip1 = x[0]
    centers = x[1][:,0]
    print(ip1.shape)
    print(centers.shape)
    sq_dis = K.square(ip1-centers)
    print(sq_dis.shape)
    l2_loss_temp = K.sum(sq_dis,1,keepdims=True)
    print(l2_loss_temp.shape)
    return l2_loss_temp

def msml_loss(y_true, y_pred):
    SN = 3
    PN = 64
    #print(y_true,y_pred)
    feat_num = SN*PN # images num
    y_pred = K.l2_normalize(y_pred,axis=1)
    feat1 = K.tile(K.expand_dims(y_pred,axis = 0),[feat_num,1,1])
    feat2 = K.tile(K.expand_dims(y_pred,axis = 1),[1,feat_num,1])
    delta = feat1 - feat2
    dis_mat = K.sum(K.square(delta),axis = 2) + K.epsilon() # Avoid gradients becoming NAN
    dis_mat = K.sqrt(dis_mat)
    positive = dis_mat[0:SN,0:SN]
    negetive = dis_mat[0:SN,SN:]
    for i in range(1,PN):
        positive = tf.concat([positive,dis_mat[i*SN:(i+1)*SN,i*SN:(i+1)*SN]],axis = 0)
        if i != PN-1:
            negs = tf.concat([dis_mat[i*SN:(i+1)*SN,0:i*SN],dis_mat[i*SN:(i+1)*SN, (i+1)*SN:]],axis = 1)
        else:
            negs = tf.concat(dis_mat[i*SN:(i+1)*SN, 0:i*SN],axis = 0)
        negetive = tf.concat([negetive,negs],axis = 0)
    positive = K.max(positive)
    negetive = K.min(negetive)
    a1 = 0.6
    loss = K.mean(K.maximum(0.0,positive-negetive+a1))
    return loss

if isCenterloss:
  lambda_c = 0.5
  input_target = Input(shape=(1,)) # single value ground truth labels as inputs
  centers = Embedding(num_classes,2)(input_target)
  #l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]),1,keepdims=True),name='l2_loss')([ip1,centers])
  l2_loss = Lambda(l2_loss_fun)([ip1, centers])
  model_centerloss = Model(inputs=[inputs,input_target],outputs=[ip2,l2_loss])#lambda y_true,y_pred: y_pred
  model_centerloss.compile(optimizer=SGD(lr=0.05), loss=["categorical_crossentropy",msml_loss ],loss_weights=[1,lambda_c],metrics=['mae','accuracy'])

# prepare callback
histories = TYY_callbacks.Histories(isCenterloss)

# fit
if isCenterloss:
  random_y_train = np.random.rand(x_train.shape[0],1)
  random_y_test = np.random.rand(x_test.shape[0],1)
  # model 概要
  model_centerloss.summary()
  model_centerloss.fit([x_train,y_train_value], [y_train, random_y_train], batch_size=batch_size, epochs=epochs, verbose=1, validation_data=([x_test,y_test_value], [y_test,random_y_test]), callbacks=[histories])

else:
  model.summary()
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test,y_test), callbacks=[histories])

