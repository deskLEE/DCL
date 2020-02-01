# ===================================================================== #
# File name:           Network_in_Network.py
# Author:              BIGBALLON
# Date update:         07/28/2017
# Python Version:      3.5.2
# Tensorflow Version:  1.2.1
# Description: 
#     Implement Network in Network(only use tensorflow) 
#     Paper Link: (Network In Network) https://arxiv.org/abs/1312.4400
#      Trick Used:
#         Data augmentation parameters
#         Color normalization
#         Weight Decay
#         Weight initialization
#         Use Nesterov momentum
# Dataset:             Cifar-10
# Testing accuracy:    91.18% - 91.25%
# ===================================================================== #

import tensorflow as tf
import pywt
from data_utility import *
import numpy as np
import cv2
import math
#import cv2.cv
import numpy as np
from matplotlib import pyplot as plt
global valflag
valflag=0
#######################
ph1=32
final_node=ph1*ph1
compression_nodenum=ph1*ph1-0
compensate_node=0
ph2=ph1
global obmatrix
obmatrix = 1 * np.random.randn(32*32,compression_nodenum) +0
obmatrix=obmatrix.astype("float32")
tf.reset_default_graph()
'''
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log_save_path', './nin_logs', 'Directory where to save tensorboard log')
tf.app.flags.DEFINE_string('model_save_path', './model/', 'Directory where to save model weights')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_integer('iteration', 391, 'iteration')
tf.app.flags.DEFINE_float('weight_decay', 0.0001, 'weight decay')
tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout')
tf.app.flags.DEFINE_float('epochs', 6000, 'epochs')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum')
'''
##3 dim tofether compress___prove this will kill the feature
def DFT_Gray(input):
    shape0=input.shape[0]
    for i in range(shape0):
        for j in range(3):
            img=input[i][:,:,j]
            #pimg = input[i].astype(np.uint8)
            #imgGray = cv2.cvtColor(pimg, cv2.COLOR_BGR2GRAY)
            #img=imgGray
            img1 = img.astype('float')
            C_temp = np.zeros(img.shape)
            dst = np.zeros(img.shape) 
            m, n = img.shape
            N = n
            C_temp[0, :] = 1 * np.sqrt(1/N)
     
            for i1 in range(1, m):
                for j1 in range(n):
                    C_temp[i1, j1] = np.cos(np.pi * i1 * (2*j1+1) / (2 * N )) * np.sqrt(2 / N ) 
                    #to set high frequency to 0
                    
            dst = np.dot(C_temp , img1)
            
            dst = np.dot(dst, np.transpose(C_temp))
            #print(dst)
            for i1 in range(1, m):
                for j1 in range(n):
                    if abs(dst[i1][j1])<=18:
                        dst[i1][j1]=0
            dst1= np.log(abs(dst))  #进行log处理
            #print(dst1.shape,input[i][:,:,j].shape)
     
           
            input[i][:,:,j]=dst
    return input
        
        
def basic_CS(input):
    shape0=input.shape[0]
    input_resize=input.reshape((shape0,-1))
    out=np.zeros((shape0,compression_nodenum))
    for i in range(shape0):
        out[i]=np.matmul(input_resize[i],obmatrix)/50
    output_resize=out.reshape((shape0,ph1,ph1,3))
    return output_resize

def CS_1dim(input):
    shape0=input.shape[0]
    input_resize=input.reshape((shape0,-1,3))
    out=np.zeros((shape0,compression_nodenum,3))
    for i in range (shape0):
        for j in range(3):
            #print(input_resize[i][:,j].shape)
            out[i][:,j]=np.matmul(input_resize[i][:,j],obmatrix)/50
    out_resize=out.reshape((shape0,ph1,ph1,3))
    print(input[1])
    print(out_resize[1])
    return out_resize
            
# ========================================================== #
# ├─ conv()
# ├─ activation(x)
# ├─ max_pool()
# └─ global_avg_pool()
# ========================================================== #

def conv(x, shape, use_bias=True, std=0.05):
    random_initializer = tf.random_normal_initializer(stddev=std)
    W = tf.get_variable('weights', shape=shape, initializer=random_initializer)
    b = tf.get_variable('bias', shape=[shape[3]], initializer=tf.zeros_initializer)
    x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    if use_bias:
        x = tf.nn.bias_add(x,b)
    return x

def activation(x):
    return tf.nn.relu(x) 

def max_pool(input, k_size=3, stride=2):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding='SAME')

def global_avg_pool(input, k_size=1, stride=1):
    return tf.nn.avg_pool(input, ksize=[1,k_size,k_size,1], strides=[1,stride,stride,1], padding='VALID')

def learning_rate_schedule(epoch_num):
      if epoch_num < 81:
          return 0.05
      elif epoch_num < 121:
          return 0.01
      else:
          return 0.001
def filter_image(input):
    shape0=input.shape[0]
    for i in range(shape0):
        for j in range(3):
            input[i][:,:,j]= cv2.adaptiveThreshold(input[i][:,:,j].astype(np.uint8),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2).astype(np.float32)
    return input
def canny(input):
    print("canny start")
    shape0=input.shape[0]
    for i in range(shape0):
        pimg = input[i].astype(np.uint8)
        imgGray = cv2.cvtColor(pimg, cv2.COLOR_BGR2GRAY)
        can=cv2.Canny(imgGray,200,300)
        for j in range (3):
            #input[i][:,:,j]=imgGray.astype(np.float32)
            input[i][:,:,j]=cv2.Canny(pimg[:,:,j],200,300).astype(np.float32)
    print("canny finished")
    return input 
def wavelet(input):
    shape0=input.shape[0]
    for i in range(shape0):
        data=input[i]
        newoutput=np.zeros(input.shape)
        cA, (cH, cV, cD)=pywt.dwt2(data, wavelet='db2', mode='symmetric', axes=(-2, -1))
        for i1 in range (cA.shape[0]):
            for i2 in range(cA.shape[1]):
                for i3 in range(cA.shape[2]):
                    newoutput[i1][i2][i3]=cA[i1][i2][i3]
    return newoutput

def DFT_cifar_data():
    train_x, train_y, test_x, test_y = prepare_data()
    train_x=DFT_Gray(train_x)
    test_x=DFT_Gray(test_x)
    '''
    if option==0:
        return train_x[:,:,:,0]
    if option==1:
        return train_y
    if option==2:
        return text_x[:,:,:,0]
    if option==3:
        return test_y
    '''
    return [train_x[:,:,:,0],train_y,test_x[:,:,:,0],test_y]
def main(_):
    global valflag
    train_x, train_y, test_x, test_y = prepare_data()
    '''
    can = cv2.Canny(train_x[1],200,300)
    cv2.imshow('candy',can)


    cv2.waitKey()
    cv2.destroyAllWindows()
    
    pimg2 = train_x[1].astype(np.uint8)
    plt.subplot(1,3,1), plt.imshow(pimg2), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    imgGray = cv2.cvtColor(pimg2, cv2.COLOR_BGR2GRAY)
    can = cv2.Canny(imgGray,200,300)
    cv2.imshow('candy',can)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    imgAdapt = cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    threshold,imgOtsu = cv2.threshold(imgGray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #plt.subplot(1,2,1), plt.imshow(pimg2), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,2), plt.imshow(imgAdapt,cmap = 'gray'), plt.title('Adaptive Gaussian Threshold'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,3), plt.imshow(imgOtsu,cmap = 'gray'), plt.title('Otsu Method'), plt.xticks([]), plt.yticks([])
    '''
    #train_x=canny(train_x)
    #test_x=canny(test_x)
    train_x=DFT_Gray(train_x)
    test_x=DFT_Gray(test_x)
    #train_x=wavelet(train_x)
    #test_x=wavelet(test_x)
    print(train_x[1])
    print(np.mean(train_x[:,:,:,0]),np.std(train_x[:,:,:,0]))
    print(np.mean(train_x[:,:,:,1]),np.std(train_x[:,:,:,1]))
    #train_x, test_x = color_preprocessing(train_x, test_x)
    #train_x=CS_1dim(train_x)
    #test_x=CS_1dim(test_x)
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_x[:,:,:,0] = (train_x[:,:,:,0] - np.mean(train_x[:,:,:,0])) / np.std(train_x[:,:,:,0])
    train_x[:,:,:,1] = (train_x[:,:,:,1] - np.mean(train_x[:,:,:,1])) / np.std(train_x[:,:,:,1])
    train_x[:,:,:,2] = (train_x[:,:,:,2] - np.mean(train_x[:,:,:,2])) / np.std(train_x[:,:,:,2])
    test_x[:,:,:,0] = (test_x[:,:,:,0] - np.mean(test_x[:,:,:,0])) / np.std(test_x[:,:,:,0])
    test_x[:,:,:,1] = (test_x[:,:,:,1] - np.mean(test_x[:,:,:,1])) / np.std(test_x[:,:,:,1])
    test_x[:,:,:,2] = (test_x[:,:,:,2] - np.mean(test_x[:,:,:,2])) / np.std(test_x[:,:,:,2])
    
    #train_x=basic_CS(train_x)
    #test_x=basic_CS(test_x)
    print(train_x.shape,test_x.shape)
    print("finaltrain_x:", train_x[1])
    
    

    # define placeholder x, y_ , keep_prob, learning_rate
    with tf.name_scope('input'):
        x  = tf.placeholder(tf.float32,[None, image_size, image_size, 3], name='input_x')
        y_ = tf.placeholder(tf.float32, [None, class_num], name='input_y')
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32)
    with tf.name_scope('learning_rate'):
        learning_rate = tf.placeholder(tf.float32)

    # build_network

    with tf.variable_scope('conv1'):
        output = conv(x,[5, 5, 3, 192],std=0.01)
        output = activation(output)

    with tf.variable_scope('mlp1-1'):
        output = conv(output,[1, 1, 192, 160])
        output = activation(output)

    with tf.variable_scope('mlp1-2'):
        output = conv(output,[1, 1, 160, 96])
        output = activation(output)

    with tf.name_scope('max_pool-1'):
        output  = max_pool(output, 3, 2)

    with tf.name_scope('dropout-1'):
        output = tf.nn.dropout(output,keep_prob)

    with tf.variable_scope('conv2'):
        output = conv(output,[5, 5, 96, 192])
        output = activation(output)

    with tf.variable_scope('mlp2-1'):
        output = conv(output,[1, 1, 192, 192])
        output = activation(output)

    with tf.variable_scope('mlp2-2'):
        output = conv(output,[1, 1, 192, 192])
        output = activation(output)

    with tf.name_scope('max_pool-2'):
        output  = max_pool(output, 3, 2)

    with tf.name_scope('dropout-2'):
        output = tf.nn.dropout(output,keep_prob)

    with tf.variable_scope('conv3'):
        output = conv(output,[3, 3, 192, 192])
        output = activation(output)

    with tf.variable_scope('mlp3-1'):
        output = conv(output,[1, 1, 192, 192])
        output = activation(output)

    with tf.variable_scope('mlp3-2'):
        output = conv(output,[1, 1, 192, 10])
        output = activation(output)

    with tf.name_scope('global_avg_pool'):
        output  = global_avg_pool(output, 8, 1)

    with tf.name_scope('moftmax'):
        output  = tf.reshape(output,[-1,10])

    # loss function: cross_entropy
    # weight decay: l2 * WEIGHT_DECAY
    # train_step: training operation

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))

    with tf.name_scope('l2_loss'):
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

    with tf.name_scope('train_step'):
        train_step = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum,use_nesterov=True).minimize(cross_entropy + l2 * FLAGS.weight_decay)

    with tf.name_scope('prediction'):
        correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    # initial an saver to save model
    saver = tf.train.Saver()
    
    # for testing
    def run_testing(sess):
        acc = 0.0
        loss = 0.0
        pre_index = 0
        add = 1000
        for it in range(10):
            batch_x = test_x[pre_index:pre_index+add]
            batch_y = test_y[pre_index:pre_index+add]
            pre_index = pre_index + add
            loss_, acc_  = sess.run([cross_entropy,accuracy],feed_dict={x:batch_x, y_:batch_y, keep_prob: 1.0})
            loss += loss_ / 10.0
            acc += acc_ / 10.0
        summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=loss), 
                                tf.Summary.Value(tag="test_accuracy", simple_value=acc)])
        return acc, loss, summary

    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(FLAGS.log_save_path,sess.graph)

        # epoch = 164 
        # batch size = 128
        # iteration = 391
        # we should make sure [bath_size * iteration = data_set_number]
        for ep in range(1,FLAGS.epochs+1):
                lr = learning_rate_schedule(ep)
                pre_index = 0
                train_acc = 0.0
                train_loss = 0.0
                start_time = time.time()
    
                print("\nepoch %d/%d:" %(ep,FLAGS.epochs))
    
                for it in range(1,FLAGS.iteration+1):
                    if pre_index+FLAGS.batch_size < 50000:
                        batch_x = train_x[pre_index:pre_index+FLAGS.batch_size]
                        batch_y = train_y[pre_index:pre_index+FLAGS.batch_size]
                    else:
                        batch_x = train_x[pre_index:]
                        batch_y = train_y[pre_index:]
    
    
                    batch_x = data_augmentation(batch_x)
    
                    _, batch_loss = sess.run([train_step, cross_entropy],feed_dict={x:batch_x, y_:batch_y, keep_prob: FLAGS.dropout, learning_rate: lr})
                    batch_acc = accuracy.eval(feed_dict={x:batch_x, y_:batch_y, keep_prob: 1.0})
    
                    train_loss += batch_loss
                    train_acc  += batch_acc
                    pre_index  += FLAGS.batch_size
    
                    if it == FLAGS.iteration:
                        train_loss /= FLAGS.iteration
                        train_acc /= FLAGS.iteration
                        
                        train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss), 
                                              tf.Summary.Value(tag="train_accuracy", simple_value=train_acc)])
    
                        val_acc, val_loss, test_summary = run_testing(sess)
    
                        summary_writer.add_summary(train_summary, ep)
                        summary_writer.add_summary(test_summary, ep)
                        summary_writer.flush()
                        if val_acc>valflag:
                            valflag=val_acc
                        print("highest test_accu",valflag)
                        print("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f" %(it, FLAGS.iteration, int(time.time()-start_time), train_loss, train_acc, val_loss, val_acc))
                    else:
                        print("highest test_accu",valflag)
                        print("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f" %(it, FLAGS.iteration, train_loss / it, train_acc / it) , end='\r')
    
        save_path = saver.save(sess, FLAGS.model_save_path)
        print("Model saved in file: %s" % save_path)      

if __name__ == '__main__':
    tf.app.run()

    



          
