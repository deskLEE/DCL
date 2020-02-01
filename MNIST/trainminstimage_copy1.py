# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:07:18 2017

@author: tyf
"""

#!/usr/bin/python3.5  
# -*- coding: utf-8 -*-    
  
import os  
import time
import numpy as np  
import tensorflow as tf  
  
from PIL import Image  
from numpy import random
import input_data
#input_count=9994+5000
#mnist = input_data.read_data_sets("D:/Fudan/compressed_learning/4-class-boatsensor/images/4class/justgray", one_hot=True)
input_count=55000
mnist=input_data.read_data_sets("D:/Fudan/thesis/CODE/COMPRESSED_LEARNING_THESIS/mnist/MNIST_data", one_hot=True)
compression_nodenum=81
obmatrix = 1 * np.random.randn(784,compression_nodenum) +0
obt = obmatrix.T
np.save("obmatrix28",obmatrix)
print ("压缩率为")
print(compression_nodenum/784)

def straight_cs(input,obmatrix1,image_num,after_nodes):
    output=np.array([[0]*after_nodes for i in range(image_num)])
    for i in range(image_num):
      output[i]=np.matmul(input[i], obmatrix1)
    return output
def straight_cs1(input,obmatrix1,image_num,after_nodes):
    output=np.array([[0]*after_nodes for i in range(image_num)])
    for i in range(image_num):
      output[i]=np.matmul( obmatrix1,input[i])
    return output
def pre_process(input,image_num,node_num):
    print(input[1])
    for i in range(image_num):
        for j in range(node_num):
            input[i][j]=1-input[i][j]
            if input[i][j]<0.15:
                input[i][j]=0
            input[i][j]=round(input[i][j],2)
    print(input[1])
    return input
        
def DFT_GRAY(input,ph1,ph2):
    #print(input.shape)
    shape0=input.shape[0]
    for i in range(shape0):
        img=np.zeros((ph1,ph2))
        img=input[i].reshape((ph1,ph2))      
        m, n = img.shape
        N = n
        C_temp = np.zeros(img.shape)
        C_temp[0, :] = 1 * np.sqrt(1/N)
         
        for i in range(1, m):
             for j in range(n):
                  C_temp[i, j] = np.cos(np.pi * i * (2*j+1) / (2 * N )
        ) * np.sqrt(2 / N )
         
        dst = np.dot(C_temp , img)
        dst = np.dot(dst, np.transpose(C_temp))
        #print(dst)
        '''
        for i1 in range(img.shape[0]):
            for j1 in range(img.shape[1]):
                if abs(dst[i1][j1])<=0.02:
                    dst[i1][j1]=0
        '''   
        dst1= np.log(abs(dst))  #进行log处理
        input[i]=dst1.reshape(ph1*ph2)
        print(dst)
    return input
 
        
#input1=pre_process(mnist.train.images,9994,784)
#input2=pre_process(mnist.validation.images,5000,784)
#test=pre_process(mnist.test.images,1705,784)
#input1=DFT_GRAY(mnist.train.images,28,28)
#input2=DFT_GRAY(mnist.validation.images,28,28)
#test=DFT_GRAY(mnist.test.images,28,28)
input1=straight_cs(mnist.train.images,obmatrix,55000,compression_nodenum)
#print(input1.shape)
input2=straight_cs(mnist.validation.images,obmatrix,5000,compression_nodenum)
test=straight_cs(mnist.test.images,obmatrix,10000,compression_nodenum)


input1=straight_cs(input1,obmatrix.T,55000,784)
#print(input1.shape)
input2=straight_cs(input2,obmatrix.T,5000,784)
test=straight_cs(test,obmatrix.T,10000,784)


#设置压缩参数

  
# 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)  
x = tf.placeholder(tf.float32, shape=[None, 784],name="x")  
y_ = tf.placeholder(tf.float32, shape=[None, 10],name="y_")  
  
x_image = tf.reshape(x, [-1,28,28, 1])  
#print(x_image)  
# 定义第一个卷积层的variables和ops  
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))  
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))  
  
L1_conv = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')  
L1_relu = tf.nn.relu(L1_conv + b_conv1)  
L1_pool = tf.nn.max_pool(L1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
  
# 定义第二个卷积层的variables和ops  
W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1))  
b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]))  
  
L2_conv = tf.nn.conv2d(L1_pool, W_conv2, strides=[1, 1, 1, 1], padding='SAME')  
L2_relu = tf.nn.relu(L2_conv + b_conv2)  
L2_pool = tf.nn.max_pool(L2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
  
  
# 全连接层  
W_fc1 = tf.Variable(tf.truncated_normal([7*7* 32, 512], stddev=0.1))  
b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]))  
  
h_pool2_flat = tf.reshape(L2_pool, [-1, 7*7*32])  
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  
  
  
# dropout  
keep_prob = tf.placeholder(tf.float32,name = "keep_prob")  
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  
  
  
# readout层  
W_fc2 = tf.Variable(tf.truncated_normal([512, 10], stddev=0.1))  
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))  
  
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2  
  
# 定义优化器和训练op  
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))  
train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)  
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")  

#代码持久化
saver=tf.train.Saver() 
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())  
  
    #print ("一共读取了 %s 个测试图像， %s 个标签" % (input_count, input_count))  
  
    # 设置每次训练op的输入个数和迭代次数，这里为了支持任意图片总数，定义了一个余数remainder，譬如，如果每次训练op的输入个数为60，图片总数为150张，则前面两次各输入60张，最后一次输入30张（余数30）  
    batch_size = 64
    iterations = 200000
    train_num=55000
    batches_count = int( train_num/ batch_size) 
    batches_count2= int( 5000/ batch_size)
    remainder = train_num % batch_size  
    print ("数据集分成 %s 批, 前面每批 %s 个数据，最后一批 %s 个数据" % (batches_count+1, batch_size, remainder))  
  
    # 执行训练迭代  
    for it in range(iterations):  
        print (time.time())
        # 这里的关键是要把输入数组转为np.array  
        for n in range(batches_count):  
            train_step.run(feed_dict={x: input1[n*batch_size:(n+1)*batch_size], y_: mnist.train.labels[n*batch_size:(n+1)*batch_size], keep_prob: 0.5})
        for n in range(batches_count2):  
            train_step.run(feed_dict={x: input2[n*batch_size:(n+1)*batch_size], y_: mnist.validation.labels[n*batch_size:(n+1)*batch_size], keep_prob: 0.5})  
        if remainder > 0:  
            start_index = batches_count * batch_size;  
            train_step.run(feed_dict={x: input1[start_index:input_count-1], y_: mnist.train.labels[start_index:input_count-1], keep_prob: 0.5})  
  
        # 每完成五次迭代，判断准确度是否已达到100%，达到则退出迭代循环  
        iterate_accuracy = 0  
        if it%5 == 0:  
            iterate_accuracy = accuracy.eval(feed_dict={x: test, y_: mnist.test.labels, keep_prob: 1.0})  
            print ('iteration %d: accuracy %s' % (it, iterate_accuracy))  
            if iterate_accuracy > 0.982: 
                print(time.time())
                break;  
    saver.save(sess,"./model/model28.ckpt")
    writer = tf.summary.FileWriter("./model/boardtest",sess.graph)
    print ('完成训练!')  