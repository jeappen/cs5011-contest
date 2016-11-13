#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 22:05:14 2016

@author: yokian
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.python.ops import rnn, rnn_cell


def batchify(lst,n):
    return [ lst[i::n] for i in xrange(n) ]

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    #hidden layer 3
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

def learn_mlp(X_train,Y_train,X_test, Y_test = None, learning_rate = 0.003
              , training_epochs = 100, batch_size = 100, display_step = 2
              , n_hidden_1 = 2000 # 1st layer number of features
              , n_hidden_2 = 200 ,n_hidden_3 = 50,beta = [1e-6,1e-6,1e-6,1e-6] # 2nd layer number of features
              , weights=None,biases = None):
    #    print 'poop'
    #
    ## Parameters
    #learning_rate = 0.01
    #training_epochs = 100
    #batch_size = 100
    #display_step = 2
    
    # Network Parameters
    
    n_input = np.shape(X_train)[1] #  data input
    n_classes = np.shape(Y_train)[1]# total classes
    n_train = np.shape(Y_train)[0]
    
    total_batch = int(n_train/batch_size)
    
    X_batches = batchify(X_train,total_batch)
    Y_batches = batchify(Y_train,total_batch)
    
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    
    """
    Now start training the classifier
    """
        
    # Store layers weight & bias
    if weights is None:
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
            'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
        }
    if biases is None:
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([n_hidden_3])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
    
    # Construct model
    pred = multilayer_perceptron(x, weights, biases)
    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y) +beta[3]*tf.nn.l2_loss(weights['out'])\
                          +beta[0]*tf.nn.l2_loss(weights['h1']))#+\
#                        beta[1]*tf.nn.l2_loss(weights['h2'])+\
#                        beta[2]*tf.nn.l2_loss(weights['h3']) +beta[3]*tf.nn.l2_loss(weights['out']) )
#    beta*tf.nn.l2_loss(biases['b1']) +\
#    beta*tf.nn.l2_loss(weights['out']) +\
#    beta*tf.nn.l2_loss(biases['out']))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initializing the variables
    init = tf.initialize_all_variables()
    
    epoch_list=[]
    accuracy_list=[]
    
    config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
    # Launch the graph
    with tf.Session(config=config) as sess:
        sess.run(init)
        # Training cycle
        prediction=tf.argmax(pred,1)
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_train/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = (X_batches[i], Y_batches[i])
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print "Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost)
                if Y_test is None:
                    best = sess.run([prediction],{x: X_test})
                else:            
                    best = sess.run([prediction],{x: X_test,y: Y_test})
                    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    print "Accuracy of MLP:", accuracy.eval({x: X_test, y: Y_test})
                    epoch_list = epoch_list + [epoch]
                    accuracy_list = accuracy_list +[accuracy.eval({x: X_test, y: Y_test})] 
        print "Optimization Finished!"
        prediction=tf.argmax(pred,1)
        if Y_test is None:
            best = sess.run([prediction],{x: X_test})
        else:            
            best = sess.run([prediction],{x: X_test,y: Y_test})
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print "Accuracy of MLP:", accuracy.eval({x: X_test, y: Y_test})
        # Test model
        
        # Calculate accuracy
        
    return (best[0],epoch_list,accuracy_list,weights,biases)

num_tests=10
e_list =[[] for i in xrange(num_tests)]
a_list =[[] for i in xrange(num_tests)]
         
         
Y_predicted0,e_list[0],a_list[0],weights1,biases1= learn_mlp(X_train,Y_train,X_test,Y_test,learning_rate = 0.003\
              , training_epochs = 10, batch_size = 2^9,beta = 1e-10*np.array([1,1,1,1]),weights=weights,biases=biases) # beta = [1e-7,1e-7,1e-7,1e-7]
#Y_predicted1,e_list[1],a_list[1],weights1,biases1 = learn_mlp(X_train,Y_train,X_test,Y_test,learning_rate = 0.003\
#              , training_epochs = 2000, batch_size = 2^10,beta = 1e-9*np.array([1,1,1,1]) ) # beta = [1e-7,1e-7,1e-7,1e-7]
#Y_predicted2,e_list[2],a_list[2] = learn_mlp(X_train,Y_train,X_test,Y_test,learning_rate = 0.003\
#              , training_epochs = 1000, batch_size = 100,beta = 0*np.array([1,1,1,1]) ) # beta = [1e-7,1e-7,1e-7,1e-7]
#Y_predicted3,e_list[3],a_list[3] = learn_mlp(X_train,Y_train,X_test,Y_test,learning_rate = 0.003\
#              , training_epochs = 1000, batch_size = 50,beta = 0*np.array([1,1,1,1]) ) # beta = [1e-7,1e-7,1e-7,1e-7]
#Y_predicted4,e_list[4],a_list[4] = learn_mlp(X_train,Y_train,X_test,Y_test,learning_rate = 0.001\
#              , training_epochs = 10000, batch_size = 100,beta = 1e-11*np.array([1,1,1,1]) ) # beta = [1e-7,1e-7,1e-7,1e-7]
#Y_predicted5,e_list[5],a_list[5] = learn_mlp(X_train,Y_train,X_test,Y_test,learning_rate = 0.003\
#              , training_epochs = 5000, batch_size = 100,beta = 1e-10*np.array([1,1,1,1]) ) # beta = [1e-7,1e-7,1e-7,1e-7]
#print 'reached 7'
#Y_predicted6,e_list[6],a_list[6] = learn_mlp(X_train,Y_train,X_test,Y_test,learning_rate = 0.003\
#              , training_epochs = 5000, batch_size = 10,beta = 1e-9*np.array([1,1,1,1]) ) # beta = [1e-7,1e-7,1e-7,1e-7]
#Y_predicted7,e_list[7],a_list[7] = learn_mlp(X_train,Y_train,X_test,Y_test,learning_rate = 0.001\
#              , training_epochs = 10000, batch_size = 25,beta = 0*np.array([1,1,1,1]) ) # beta = [1e-7,1e-7,1e-7,1e-7]
##Y_predicted8,e_list[0],a_list[0] = learn_mlp(X_train,Y_train,X_test,Y_test,learning_rate = 0.003\
#              , training_epochs = 20, batch_size = 100,beta = 1e-9*np.array([1,1,1,1]) ) # beta = [1e-7,1e-7,1e-7,1e-7]
#Y_predicted9,e_list[0],a_list[0] = learn_mlp(X_train,Y_train,X_test,Y_test,learning_rate = 0.003\
#              , training_epochs = 4000, batch_size = 100,beta = 1e-9*np.array([1,1,1,1]) ) # beta = [1e-7,1e-7,1e-7,1e-7]

#Y_predicted = learn_mlp(X_train[:,-100:-1],Y_train,X_test[:,-100:-1],Y_test,training_epochs = 100,beta = 1e-6)
#pca = PCA(n_components=X_train.shape[1])
#pca.fit(X_train)  
#print(pca.explained_variance_ratio_) 
#T_test = pca.transform(X_test)
#T_train = pca.transform(X_train)
#Y_predicted = learn_mlp(T_train[:,:2500],Y_train,T_test[:,:2500],Y_test,training_epochs = 100,beta = 1e-6)
