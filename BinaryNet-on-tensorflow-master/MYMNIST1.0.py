# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:53:25 2018

@author: tyf
"""

import tensorflow as tf
import binary_layer 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from mnist import download_mnist

input_count=55000
final_node=400
compression_nodenum=388 
compensate_node=12
ph1=20
ph2=20

obmatrix = 1 * np.random.randn(784,compression_nodenum) +0
obmatrix=obmatrix.astype("float32")
print ("压缩率为")
print(compression_nodenum/784)

def straight_cs(input,obmatrix,image_num,after_nodes):
    output=np.array([[0]*after_nodes for i in range(image_num)])
    for i in range(image_num):
      print(input[i].shape,obmatrix.shape)
      output[i]=np.matmul(input[i], obmatrix)
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

#返回大的数值
def bidaxiao(a,b):
    if (a>=b):
        return a
    else:
        return b
    
                   
def plan12(input,compensate_node):
    shape0=input.shape[0]
    output=np.zeros([shape0,(compensate_node+compression_nodenum)])
    #rownum=compensate_node/3
    temp_compressedonly=np.zeros([1,compression_nodenum+compensate_node])
    for i in range(shape0):
        temp_compressedonly=np.matmul(input[i],obmatrix)    #yasuoganzhi
        for j in range(compression_nodenum):
            output[i][j]=temp_compressedonly[j]
        temp=np.zeros((4,int(compensate_node/4)))
        #temp_resize=np.zeros((1,compensate_node))
        temp=input[i].reshape((4,-1))
        imager=input[i].reshape(4,-1)
        for j in range(4):
            maxkeep1=0
            maxkeep0=0
            changes=0
            temp_maxkeep0=0
            temp_maxkeep1=0
            for k in range(1 ,-1):
                if (imager[j][k]==imager[j][k-1]):
                    if(imager[j][k]==0):
                        
                        temp_maxkeep0=temp_maxkeep0+1
                        maxkeep0=bidaxiao(maxkeep0,temp_maxkeep0)
                    else:
                        temp_maxkeep1=temp_maxkeep1+1
                        maxkeep1=bidaxiao(maxkeep1,temp_maxkeep1)
                else:
                     temp_maxkeep0=0
                     temp_maxkeep1=0
                     changes=changes+1
            temp[j][0]=maxkeep0
            temp[j][1]=maxkeep1
            temp[j][2]=changes
            temp_resize=temp.reshape((1,-1))
        for j in range (compensate_node):
            output[i][j+compression_nodenum]=temp_resize[0][j]
    ##print(output.shape)
    return output

def fully_connect_bn(pre_layer, output_dim, act, use_bias, training):
    pre_act = binary_layer.dense_binary(pre_layer, output_dim,
                                    use_bias = use_bias,
                                    activation = None,
                                    kernel_constraint = lambda w: tf.clip_by_value(w, -1.0, 1.0))
    bn = binary_layer.batch_normalization(pre_act, momentum=0.9, epsilon=1e-4, training=training)
    if act == None:
        output = bn
    else:
        output = act(bn)
    return output

def no_scale_dropout(pre_layer, drop_rate, training):
    drop_layer = tf.layers.dropout(pre_layer, rate=drop_rate, training=training)
    #return tf.cond(training, lambda: drop_layer*(1-drop_rate), lambda: drop_layer)
    return drop_layer

# A function which shuffles a dataset
def shuffle(X,y):
    print(len(X))
    shuffle_parts = 1
    chunk_size = int(len(X)/shuffle_parts)
    shuffled_range = np.arange(chunk_size)

    X_buffer = np.copy(X[0:chunk_size])
    y_buffer = np.copy(y[0:chunk_size])

    for k in range(shuffle_parts):

        np.random.shuffle(shuffled_range)

        for i in range(chunk_size):

            X_buffer[i] = X[k*chunk_size+shuffled_range[i]]
            y_buffer[i] = y[k*chunk_size+shuffled_range[i]]

        X[k*chunk_size:(k+1)*chunk_size] = X_buffer
        y[k*chunk_size:(k+1)*chunk_size] = y_buffer

    return X,y

# This function trains the model a full epoch (on the whole dataset)
def train_epoch(X, y, sess, batch_size=32):
    batches = int(len(X)/batch_size)
    for i in range(batches):
        sess.run([train_kernel_op, train_other_op],
            feed_dict={ x: X[i*batch_size:(i+1)*batch_size],
                        target: y[i*batch_size:(i+1)*batch_size],
                        training: True})
def conv_bn(pre_layer, kernel_num, kernel_size, padding, activation, training, epsilon=1e-4, alpha=.1, binary=True, stochastic=False, H=1., W_LR_scale="Glorot"):
    conv = binary_layer.conv2d_binary(pre_layer, kernel_num, kernel_size, padding=padding, binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale)
    bn = binary_layer.batch_normalization(conv, epsilon=epsilon, momentum = 1-alpha, training=training)
    output = activation(bn)
    return output

def conv_pool_bn(pre_layer, kernel_num, kernel_size, padding, pool_size, activation, training, epsilon=1e-4, alpha=.1, binary=True, stochastic=False, H=1., W_LR_scale="Glorot"):
    conv = binary_layer.conv2d_binary(pre_layer, kernel_num, kernel_size, padding=padding, binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale)
    pool = tf.layers.max_pooling2d(conv, pool_size=pool_size, strides=pool_size)
    bn = binary_layer.batch_normalization(pool, epsilon=epsilon, momentum = 1-alpha, training=training)
    output = activation(bn)
    return output
x = tf.placeholder(tf.float32, shape=[None, final_node])
#x_image = tf.reshape(x, [-1,ph1,ph2, 1]) 
target = tf.placeholder(tf.float32, shape=[None, 10])
training = tf.placeholder(tf.bool)

download_mnist.maybe_download('./mnist/MNIST_data/')
mnist = input_data.read_data_sets('./mnist/MNIST_data/', one_hot=True)

# convert class vectors to binary class vectors
for i in range(mnist.train.images.shape[0]):
    mnist.train.images[i] = mnist.train.images[i] * 2 - 1
for i in range(mnist.test.images.shape[0]):
    mnist.test.images[i] = mnist.test.images[i] * 2 - 1
for i in range(mnist.train.labels.shape[0]):
    mnist.train.labels[i] = mnist.train.labels[i] * 2 - 1 # -1 or 1 for hinge loss
for i in range(mnist.test.labels.shape[0]):
    mnist.test.labels[i] = mnist.test.labels[i] * 2 - 1
print(mnist.test.labels.shape)
print(mnist.test.images.shape)
##########################
alpha = .1
print("alpha = "+str(alpha))
epsilon = 1e-4
print("epsilon = "+str(epsilon))

# BinaryOut
activation = binary_layer.binary_tanh_unit
print("activation = binary_net.binary_tanh_unit")

# BinaryConnect
binary = True
print("binary = "+str(binary))
stochastic = False
print("stochastic = "+str(stochastic))
# (-H,+H) are the two binary values
H = 1.
print("H = "+str(H))
W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
print("W_LR_scale = "+str(W_LR_scale))
######
#x_input=tf.layer
layer0 = no_scale_dropout(x, drop_rate=0.2, training=training)

layer1 = fully_connect_bn(layer0, 4096, act=binary_layer.binary_tanh_unit, use_bias=True, training=training)
layer1_dp = no_scale_dropout(layer1, drop_rate=0.5, training=training)

layer2 = fully_connect_bn(layer1_dp, 4096, act=binary_layer.binary_tanh_unit, use_bias=True, training=training)
layer2_dp = no_scale_dropout(layer2, drop_rate=0.5, training=training)

layer3 = fully_connect_bn(layer2_dp, 4096, act=binary_layer.binary_tanh_unit, use_bias=True, training=training)
layer3_dp = no_scale_dropout(layer3, drop_rate=0.5, training=training)

layer4 = fully_connect_bn(layer3_dp, 10, act=None, use_bias=True, training=training)

#out_act_training = tf.nn.softmax_cross_entropy_with_logits(logits=layer4, labels=target)
#out_act_testing = tf.nn.softmax(logits=layer4)
#out_act = tf.cond(training, lambda: out_act_training, lambda: out_act_testing)

loss = tf.reduce_mean(tf.square(tf.maximum(0.,1.-target*layer4)))

epochs = 1000
lr_start = 0.003
lr_end = 0.0000003
lr_decay = (lr_end / lr_start)**(1. / epochs)
global_step1 = tf.Variable(0, trainable=False)
global_step2 = tf.Variable(0, trainable=False)
lr1 = tf.train.exponential_decay(lr_start, global_step=global_step1, decay_steps=int(mnist.train.images.shape[0]/100), decay_rate=lr_decay)
lr2 = tf.train.exponential_decay(lr_start, global_step=global_step2, decay_steps=int(mnist.train.images.shape[0]/100), decay_rate=lr_decay)

sess = tf.Session()
saver = tf.train.Saver()
#saver.restore(sess, "model/model.ckpt")

other_var = [var for var in tf.trainable_variables() if not var.name.endswith('kernel:0')]
opt = binary_layer.AdamOptimizer(binary_layer.get_all_LR_scale(), lr1)
opt2 = tf.train.AdamOptimizer(lr2)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):   # when training, the moving_mean and moving_variance in the BN need to be updated.
    train_kernel_op = opt.apply_gradients(binary_layer.compute_grads(loss, opt),  global_step=global_step1)
    train_other_op  = opt2.minimize(loss, var_list=other_var,  global_step=global_step2)


accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(layer4, 1), tf.argmax(target, 1)), tf.float32))
sess.run(tf.global_variables_initializer())

old_acc = 0.0
X_train, y_train = shuffle(mnist.train.images, mnist.train.labels)
C_train=plan12(X_train, compensate_node)
C_test=plan12(mnist.test.images,compensate_node)
print("x_train.shape is" ,X_train.shape)
for i in range(epochs):
    train_epoch(C_train, y_train, sess)
    #X_train, y_train = shuffle(mnist.train.images, mnist.train.labels)
    if i % 10==0:
        hist = sess.run([accuracy, opt._lr],
                        feed_dict={
                        x: C_test,
                        target: mnist.test.labels,
                        training: False
                        })
        print("epoches:", i)    
        print(hist)

    if hist[0] > old_acc:
        old_acc = hist[0]
        save_path = saver.save(sess, "./mnist/model/model.ckpt")

'''
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=layer4))  
train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)  
correct_prediction = tf.equal(tf.argmax(target, 1), tf.argmax(target, 1))  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
#代码持久化
#saver=tf.train.Saver() 
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())  
  
    #print ("一共读取了 %s 个测试图像， %s 个标签" % (input_count, input_count))  
  
    # 设置每次训练op的输入个数和迭代次数，这里为了支持任意图片总数，定义了一个余数remainder，譬如，如果每次训练op的输入个数为60，图片总数为150张，则前面两次各输入60张，最后一次输入30张（余数30）  
    batch_size = 32
    iterations = 50000 
    train_num=55000
    batches_count = int( train_num/ batch_size) 
    batches_count2= int( 5000/ batch_size)
    remainder = train_num % batch_size  
    print ("数据集分成 %s 批, 前面每批 %s 个数据，最后一批 %s 个数据" % (batches_count+1, batch_size, remainder))  
  
    # 执行训练迭代  
    for it in range(iterations):  
        # 这里的关键是要把输入数组转为np.array  
        for n in range(batches_count):  
            train_step.run(feed_dict={x: mnist.train.images[n*batch_size:(n+1)*batch_size], target: mnist.train.labels[n*batch_size:(n+1)*batch_size]})
        for n in range(batches_count2):  
            train_step.run(feed_dict={x: mnist.validation.images[n*batch_size:(n+1)*batch_size], target: mnist.validation.labels[n*batch_size:(n+1)*batch_size]})  
        if remainder > 0:  
            start_index = batches_count * batch_size;  
            train_step.run(feed_dict={x: mnist.train.images[start_index:60000-1], target: mnist.train.labels[start_index:60000-1]})  
        print(it)
        # 每完成五次迭代，判断准确度是否已达到100%，达到则退出迭代循环  
        iterate_accuracy = 0  
        if it%50 == 0:  
            iterate_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, target: mnist.test.labels})  
            print ('iteration %d: accuracy %s' % (it, iterate_accuracy))  
            if iterate_accuracy > 0.99:  
                break;  
    #saver.save(sess,"D:/Fudan/compressed_learning/minst/model/model99.ckpt")
    print ('完成训练!')  
'''