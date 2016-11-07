# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 00:33:32 2016

ML COntest
@author: joera_000
"""

import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier,VotingClassifier,BaggingClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from tensorflow.python.ops import rnn, rnn_cell

def read_data():
    z = zipfile.ZipFile('../DATASET.zip')
    print z.namelist()
    df_train = pd.read_csv(z.open(z.namelist()[3]),header = None)
    df_train['label'] = pd.read_csv(z.open(z.namelist()[4]),header = None)
    df_ldb = pd.read_csv(z.open(z.namelist()[1]), header = None)
    return df_train,df_ldb


def batchify(lst,n):
    return [ lst[i::n] for i in xrange(n) ]

            
train_test_split_fraction = 0.3
parameter_grid = {
    'max_features': [0.5, 1.],
    'max_depth': [5., None]
}

#grid_search = GridSearchCV(RandomForestClassifier(n_estimators = 100), parameter_grid,
#                            cv=5, verbose=3)

#df = pd.read_csv('../Training_Dataset.csv')

if 'df' not in locals():
    (df, df_ldb) = read_data()
    df = pd.concat([df, pd.get_dummies(df['label'], prefix='class')], axis=1)
cols = df.columns.tolist()

output_index_start = -12
label_col_index = output_index_start-1
#bring vote to start
#cols = cols[-5:]+cols[:-5]

#df = df[cols]

#Test how correlated, sets of features are with final vote
#startnum_list = [1,6,11,16,21]
#acc_argmax = []
#colname_set = []
#for startnum in startnum_list:
#    colnames = ['mvar'+str(i+startnum) for i in xrange (len(party_dict))]
#    colname_set = colname_set + [colnames]
#    cols_party_map = dict(zip(colnames,party_names))
#    argmax_predict = df[colnames].idxmax(axis=1).map(cols_party_map).map(party_dict)
#    acc_argmax = acc_argmax+[np.sum(df[cols[0]] == argmax_predict)/(len(df)+0.0)]
#    df['addedvar'+str(startnum)] = argmax_predict

       
testcols = cols#['vote','pvp']+['addedvar'+str(startnum) for startnum in startnum_list]#+cols[30:39]#+[y for x in colname_set[:] for y in x]#colname_set[0]+colname_set[1]
                             
df_train, df_test = train_test_split(df[testcols],\
                                     test_size = train_test_split_fraction,random_state = 4)

train_data = df_train.values
test_data = df_test.values

#In case of potato PC
del df_train,df_test

Ylabel_train = train_data[:,label_col_index]
Ylabel_test = test_data[:,label_col_index]
Y_train = train_data[:,output_index_start:]
Y_test = test_data[:,output_index_start:]
X_train = train_data[:,:label_col_index] #-1 to skip last column
X_test = test_data[:,:label_col_index]



"""
Enter Tensorflow
"""
# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

def learn_mlp(X_train,Y_train,X_test, Y_test = None, learning_rate = 0.01
              , training_epochs = 100, batch_size = 100, display_step = 2
              , n_hidden_1 = 50 # 1st layer number of features
              , n_hidden_2 = 20  # 2nd layer number of features
              ):
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
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    # Construct model
    pred = multilayer_perceptron(x, weights, biases)
    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initializing the variables
    init = tf.initialize_all_variables()
    
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
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
        
    return best[0]

class RNN_c(object):
    
    def RNN(self,x, weights, biases):
    
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, self.n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, self.n_steps, x)
    
        # Define a lstm cell with tensorflow
#        with tf.variable_scope('poop', reuse =True):    
        tf.get_variable_scope().reuse_variables()
        lstm_cell = rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
    
        # Get lstm cell output
        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    
        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']
    
    def learn_RNN(self,X_train,Y_train,X_test, Y_test = None, learning_rate = 0.01
                  , training_iters = 100000, batch_size = 128, display_step = 10
                  , n_input = 50 # 1st layer number of features
                  , n_hidden = 128  # 2nd layer number of features
                  ):
    
        # Network Parameters
        # n_input = 28 # MNIST data input (img shape: 28*28)
        self.n_steps = 64 # timesteps
        self.n_hidden = n_hidden #128 # hidden layer num of features
        # n_classes = 10 # MNIST total classes (0-9 digits)
        self.n_input = np.shape(X_train)[1]/self.n_steps #  data input
        self.n_classes = np.shape(Y_train)[1]# total classes
        self.n_train = np.shape(Y_train)[0]
    
        total_batch = int(self.n_train/batch_size)
        
        X_batches = batchify(X_train, total_batch)
        Y_batches = batchify(Y_train, total_batch)
    
        # tf Graph input
        x = tf.placeholder("float", [None, self.n_steps, self.n_input])
        y = tf.placeholder("float", [None, self.n_classes])
    
        # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }
    
    
    
    
        pred = self.RNN(x, weights, biases)
    
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
        # Initializing the variables
        init = tf.initialize_all_variables()
    
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            # Keep training until reach max iterations
            while step * batch_size < training_iters:
                total_batch = int(self.n_train/batch_size)
                for i in range(total_batch):
                    batch_x, batch_y = (X_batches[i], Y_batches[i])
                    # Reshape data to get 28 seq of 28 elements
                    batch_size = batch_x.shape[0]*batch_x.shape[1]/np.shape(X_train)[1]
                    batch_x = batch_x.reshape((batch_size, self.n_steps, self.n_input))
                    # Run optimization op (backprop)
                    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                    if step % display_step == 0:
                        # Calculate batch accuracy
                        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                        # Calculate batch loss
                        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                        print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc))
                    step += 1
            print("Optimization Finished!")
            prediction=tf.argmax(pred,1)
    
            # Calculate accuracy for 128 mnist test images
            # test_len = 128
#            if Y_test is None:
#                best = sess.run([prediction],{x: X_test})
#            else:            
#                best = sess.run([prediction],{x: X_test,y: Y_test})
#                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#                print "Accuracy of MLP:", accuracy.eval({x: X_test, y: Y_test})
            test_data = X_test.reshape((-1, self.n_steps, self.n_input))
            test_label = Y_test
            print("Testing Accuracy:", \
                sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
            best = sess.run([prediction], feed_dict={x: test_data, y: test_label})
            
        return best[0]


pca = PCA(n_components=X_train.shape[1])

Y_predicted = learn_mlp(X_train,Y_train,X_test,Y_test,training_epochs = 10)

rnn_obj = RNN_c();
#Y_predicted = rnn_obj.learn_RNN(X_train,Y_train,X_test,Y_test)

clf_SET = []
clf_labels = ["RF","VotingClassifier","Bagged SVM","Grad Boosted Dec Tree","MLP"]

#
##grid_search.fit(train_data[0:,1:], train_data[0:,0])


#
##Random Forest
#model = RandomForestClassifier(n_estimators = 600, random_state=0)
#model = model.fit(train_data[:,5:-1], train_data[:,-1])
#
##VotingEnsembleMethod
#clf1 = LogisticRegression(random_state=1)
#clf2 = RandomForestClassifier(n_estimators = 600,random_state=0)
#clf3 = GaussianNB()
#
#eclf2 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft')
#eclf2 = eclf2.fit(train_data[:,5:-1], train_data[:,-1])
#
##SVM
##clf_svm = svm.SVC()
##clf_svm.fit(train_data[0:,5:-1], train_data[:,-1])
#
##Bagged SVM
#clf_svm_bag = BaggingClassifier(svm.SVC(),\
#                             max_samples=0.05, max_features=0.3)
#clf_svm_bag.fit(train_data[:,5:-1], train_data[:,-1])
##Gradient Boosting
#clf_gb = GradientBoostingClassifier(n_estimators=600, learning_rate=1.0,\
#                                    max_depth=1, random_state=0).fit(train_data[:,5:-1], train_data[:,-1])
#
##Make list here to display results
#clf_SET = [model,eclf2,clf_svm_bag,clf_gb]

"""
Now import the Leaderboard data
"""
#df_test = pd.read_csv('../Leaderboard_Dataset.csv')
##df_test = df_test.drop(['citizen_id'], axis=1)
##df_test = df_test.dropna()
#df_test = df_test.fillna(train_median)
#
#edu_median_dict = {'mvar30':edu_dict.keys()[edu_dict.values().index(df_education.median())]}
#                   
#df_test = df_test.fillna(edu_median_dict)
#                   
#all_regions_test = set(df_test['mvar32'].unique()) | set(df_test['mvar33'].unique())
#diff_regions = list(set(all_regions_test) - set(all_regions))
#prev_region_med = df.median()['prev_region']
#curr_region_med = df.median()['curr_region']
#
#diff_region_map = dict(zip(diff_regions,[prev_region_med for i in xrange(len(diff_regions))]))
#
#z = region_map.copy()
#z.update(diff_region_map)
#
#df_test['age_groups'] = df_test['mvar27'].map(amap).astype(int)
#
##df_test['education'] = df_test['mvar30'].map(custom_edu_dict).astype(int)
#
#df_test = pd.concat([df_test, pd.get_dummies(df_test['mvar30'], prefix='edu')], axis=1)
#
#df_test['prev_region'] = df_test['mvar32'].map(z).astype(int)
#df_test['curr_region'] = df_test['mvar33'].map(z).astype(int)
#df_test['pvp'] = df_test['party_voted_past'].map(party_dict).astype(int)
#df_test = df_test.drop(['party_voted_past','mvar27','mvar30','mvar32','mvar33'], axis=1)
#test_data = df_test.values


output = [[] for clf in clf_SET]
accuracy = [[] for clf in clf_SET]
for i in range(len(clf_SET)):
    output[i] = clf_SET[i].predict(test_data[:,5:-1])
    accuracy[i] = np.sum(output[i]==test_data[:,-1])/(len(output[i])+0.0)
    print "Accuracy of",clf_labels[i]," is ",accuracy[i]

"""
Okay, now print the output
"""

result = np.c_[Y_predicted]
df_result = pd.DataFrame(result)


df_result.to_csv('../results/tf_mlp_result.csv', index=False, header = None)