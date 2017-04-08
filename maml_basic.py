#
#  mnist_cnn_bn.py   date. 5/21/2016
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


num_tasks = 5
num_weights = 6
alpha = 1
beta = 1
num_epochs = 1000
eta = .01

#function which creates parameter objects
def model_variable(shape, name,type='weight',stddev=.1):
    if type == 'bias':
        variable = tf.get_variable(name=name,
                                    dtype=tf.float32,
                                    initializer=tf.zeros(shape=shape))
    else:
        variable = tf.get_variable(name=name,
                               dtype=tf.float32,
                               shape = shape,
                               initializer=tf.random_normal_initializer(stddev=stddev))
    tf.add_to_collection('model_variables', variable)
    
    return variable

w1 = model_variable([1,40],'w1')
w2 = model_variable([40,40],'w2')
b1 = model_variable([40],'b1','bias')
b2 = model_variable([40],'b2','bias')
wlast = model_variable([40,1],'wlast')
blast = model_variable([1],'blast','bias')

# Create the model
def inference(x):

    tmp = x*w1 + b1
    tmp = tf.nn.relu(tmp)

    tmp = tf.matmul(tmp,w2) + b2
    tmp = tf.nn.relu(tmp)

    tmp = tf.matmul(tmp,wlast) + blast
    y_hat = tf.nn.relu(tmp)   

    return y_hat      

def maml(x, y, weights_i):
    save_updates = []
    for i in range(num_tasks):
        g = []
        #get the gradients for the data corresponding to this task
        l = tf.square(inference(x[i]) - y[i]) 
        grads = tf.gradients(l, weights_i)
        for weight_i,grad in zip(weights_i,grads):
            save_updates.append(tf.Variable(tf.zeros(weight_i.get_shape())))
            #get the look-ahead gradient
            g.append(tf.gradients(l,weight_i - alpha * grad))

        j = 0
        for weight_i,g in zip(weights_i,grads):
            save_updates[j] = save_updates[j] - beta * grad
            j = j + 1
    updates = []
    for i, weight_i in zip(range(num_weights),weights_i):
        updates.append(weight_i.assign(weight_i - save_updates[i]))

    return tf.group(*updates)


# Variables
x_ = tf.placeholder(tf.float32, [None, 1])
y_ = tf.placeholder(tf.float32, [None, 1])


y_pred = inference(x_)

model_variables = tf.get_collection('model_variables')
loss = tf.pow(y_pred - y_,2)
optim = tf.train.GradientDescentOptimizer(eta).minimize(loss, var_list=model_variables)

run_iter = maml(x_,y_,model_variables)


#create sine wave data
def data():
    x = np.random.uniform(0,10,size=num_tasks)
    y =  np.arange(1,num_tasks+1) *  np.sin(x)
    return [x,y]

def particular_data(task,size=1): #task refers to amplitude
    x = np.random.uniform(0,10,size=size)
    y = task * np.sin(x)
    return [x,y]
#



sess = tf.Session()
sess.run(tf.global_variables_initializer())


for _ in range(num_epochs):
    batch_xs, batch_ys = data() #1 from each task
    ex = np.reshape(batch_xs,[num_tasks,1])
    targ = np.reshape(batch_ys,[num_tasks,1])
    _ = sess.run([run_iter], feed_dict={x_ : ex, y_ : targ})

num_final_steps = 10000
for _ in range(num_final_steps):
    x,y = particular_data(5) #amplitude 5
    ex = np.reshape(x,[-1,1])
    targ = np.reshape(y,[-1,1])
    l,_ = sess.run([loss,optim],feed_dict={x_ : ex, y_ : targ})

test_data = particular_data(5,1000)
ex = np.reshape(test_data[0],[-1,1])
targ = np.reshape(test_data[1],[-1,1])
y = sess.run([y_pred],feed_dict={x_ : ex, y_ : targ})
ex = np.squeeze(ex)
y = np.squeeze(y)

plt.plot(ex,y,'ro')
plt.plot(ex,targ,'go')
plt.xlabel('x')
plt.ylabel('y')
plt.title('MAML')
plt.show()