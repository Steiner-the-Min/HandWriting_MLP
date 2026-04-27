# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 20:32:32 2021

@author: zhaodf
"""
#采用MLP模型对MNIST手写数字进行识别
#MNIST是一个手写体数字的图片数据集，该数据集来由美国国家标准与技术研究所
# （National Institute of Standards and Technology (NIST)）发起整理
#一共统计了来自250个不同的人手写数字图片，其中50%是高中生，50%来自人口普查局的工作人员
#在上述数据集中，训练集一共包含了 60,000 张图像和标签，而测试集一共包含了 10,000 张图像和标签

#tensorflow1.13.1 and concon2.2.4
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
# print(keras.__version__)

from keras.datasets import mnist
import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = mnist.load_data()#加载数据

plt.subplot(241)
plt.imshow(X_train[12], cmap=plt.get_cmap('gray'))
plt.subplot(242)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(243)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(244)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
plt.subplot(245)
plt.imshow(X_train[4], cmap=plt.get_cmap('gray'))
plt.subplot(246)
plt.imshow(X_train[5], cmap=plt.get_cmap('gray'))
plt.subplot(247)
plt.imshow(X_train[6], cmap=plt.get_cmap('gray'))
plt.subplot(248)
plt.imshow(X_train[7], cmap=plt.get_cmap('gray'))

print("test")

from keras.models import Sequential # 导入Sequential模型
from keras.layers import Dense # 全连接层用Dense类
from keras.utils import np_utils # 导入np_utils是为了用one hot encoding方法将输出标签的向量（vector）转化为只在出现对应标签的那一列为1，其余为0的布尔矩阵
from keras.layers import Dropout

#数据集是3维的向量（instance length,width,height).对于多层感知机，模型的输入是二维的向量，因此这里需要将数据集reshape，即将28*28的向量转成784长度的数组。可以用numpy的reshape函数轻松实现这个过程。
num_pixels = X_train.shape[1] * X_train.shape[2] 
X_train = X_train.reshape(X_train.shape[0],num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0],num_pixels).astype('float32')

#给定的像素的灰度值在0-255，为了使模型的训练效果更好，通常将数值归一化映射到0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# 搭建神经网络模型，创建一个函数，建立含有一个隐层的神经网络
def baseline_model():
    model = Sequential()
    model.add(Dense(512, input_dim=num_pixels, activation='relu'))
    model.add(Dropout(0.2)) # 训练时随机断开20%的连接，防止依赖某些特定的像素
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()
model.summary()
history=model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=100, batch_size=200) #训练

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

#fig = plt.figure()
figsize = 7,5
figure, ax = plt.subplots(figsize=figsize)
plt.tick_params(labelsize=12)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.plot(history.history['acc'],'-r',label='training acc',linewidth=1.5)
plt.plot(history.history['val_acc'],'-b',label='val acc',linewidth=1.5)
plt.title('model accuracy',font2)
plt.ylabel('accuracy',font2)
plt.xlabel('epoch',font2)
plt.legend(loc='lower right',prop=font2)

#fig = plt.figure()
figsize = 7,5
figure, ax = plt.subplots(figsize=figsize)
plt.tick_params(labelsize=12)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.plot(history.history['loss'],'-r',label='training loss',linewidth=1.5)
plt.plot(history.history['val_loss'],'-b', label='val loss',linewidth=1.5)
plt.title('model loss',font2)
plt.ylabel('loss',font2)
plt.xlabel('epoch',font2)
plt.legend(loc='upper right',prop=font2)

figsize = 7,5
figure, ax = plt.subplots(figsize=figsize)
plt.tick_params(labelsize=12)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.plot(history.history['acc'],'-g',label='training acc',linewidth=1.5)
plt.plot(history.history['val_acc'],'-r',label='val acc',linewidth=1.5)
plt.plot(history.history['loss'],'-y',label='training loss',linewidth=1.5)
plt.plot(history.history['val_loss'],'-b', label='val loss',linewidth=1.5)
plt.title('model loss and accuracy',font2)
plt.ylabel('value',font2)
plt.xlabel('epoch',font2)
plt.legend(loc='best',prop=font2)

scores = model.evaluate(X_test,y_test) #model.evaluate 返回计算误差和准确率

print(scores)
print("Base Error:%.2f%%"%(100-scores[1]*100))

# 1. 将模型结构保存为 JSON 字符串
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# 2. 仅保存权重到 H5 文件
model.save_weights("model_weights.h5")
print("模型结构与权重已分开保存成功！")
# plt.show()