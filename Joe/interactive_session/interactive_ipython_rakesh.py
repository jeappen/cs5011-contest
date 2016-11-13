import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def read_data_full():
    f_train = open('/media/yokian/Storage/ml/DATASET/train_data', 'r')
    f_labels = open('/media/yokian/Storage/ml/DATASET/train_labels', 'r')
    f_ldb = open('/media/yokian/Storage/ml/DATASET/leaderboardTest_data', 'r')
#    print z.namelist()
    df_train = pd.read_csv(f_train,header = None,dtype=np.float32)
    df_train['label'] = pd.read_csv(f_labels,header = None)
    df_ldb = pd.read_csv(f_ldb, header = None,dtype=np.float32)
    return df_train,df_ldb
    

def batchify(lst,n):
    return [ lst[i::n] for i in xrange(n) ]

            
train_test_split_fraction = 0.4
if 'df' not in locals():
    (df, df_ldb) = read_data_full()
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
                                     test_size = train_test_split_fraction,random_state =1)

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

restore_save = False
training_epochs = 100
batch_size = 100
display_step = 2
n_hidden_1 = 2000
n_hidden_2 = 1200
n_hidden_3 = 600
num_layers =2
beta = 0
weights = None
biases = None

def multilayer_perceptron(x, weights, biases,num_layers = 1):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.elu(layer_2)
    #hidden layer 3
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Output layer with linear activation
    if num_layers == 1:
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    elif num_layers == 2:
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    else :
        out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

n_input = np.shape(X_train)[1] #  data input
n_classes = np.shape(Y_train)[1]# total classes
n_train = np.shape(Y_train)[0]

total_batch = int(n_train/batch_size)

X_batches = batchify(X_train,total_batch)
Y_batches = batchify(Y_train,total_batch)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

model_save="/home/yokian/Tensorflow/saved/mlp_contest_new.ckpt"

"""
Now start training the classifier
"""
learning_rate_val = 0.001
   
# Store layers weight & bias
if weights is None:
    print 'initialising weights'
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),name="h1"),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]),name="h2"),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]),name="h3")
    }
    if num_layers ==1:
        weights['out']=tf.Variable(tf.random_normal([n_hidden_1, n_classes]),name="hout")
    elif num_layers == 2:
        weights['out']=tf.Variable(tf.random_normal([n_hidden_2, n_classes]),name="hout")
    else:
        weights['out']=tf.Variable(tf.random_normal([n_hidden_3, n_classes]),name="hout")
    
    
if biases is None:
    print 'initialising biases'
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1]),name="b1"),
        'b2': tf.Variable(tf.random_normal([n_hidden_2]),name="b2"),
        'b3': tf.Variable(tf.random_normal([n_hidden_3]),name="b3"),
        'out': tf.Variable(tf.random_normal([n_classes]),name="bout")
    }

# Construct model
pred = multilayer_perceptron(x, weights, biases,num_layers)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)+beta*tf.nn.l2_loss(weights['h1'])+beta*tf.nn.l2_loss(weights['out']))#+\
#                        beta[1]*tf.nn.l2_loss(weights['h2'])+\
#                        beta[2]*tf.nn.l2_loss(weights['h3']) +beta[3]*tf.nn.l2_loss(weights['out']) )
#    beta*tf.nn.l2_loss(biases['b1']) +\
#    beta*tf.nn.l2_loss(weights['out']) +\
#    beta*tf.nn.l2_loss(biases['out']))
# learning_rate = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.AdamOptimizer(learning_rate_val).minimize(cost)

# Initializing the variables
if restore_save== False:
    init = tf.initialize_all_variables()

saver = tf.train.Saver()
epoch_list=[]
accuracy_list=[]

sess=tf.InteractiveSession()
sess.run(init)
print("Initialising Session...")
prediction=tf.argmax(pred,1)
#now run this separately

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_train/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_x, batch_y = (X_batches[i], Y_batches[i])
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,\
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
#                    save_path = saver.save(sess, model_save)
#                    print("Model saved in file: %s" % save_path)
print "Optimization Finished!"
prediction=tf.argmax(pred,1)
if Y_test is None:
    best = sess.run([prediction],{x: X_test})
else:            
    best = sess.run([prediction],{x: X_test,y: Y_test})
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy of MLP:", accuracy.eval({x: X_test, y: Y_test})
save_path = saver.save(sess, model_save)
print("Model saved in file: %s" % save_path)

e_count = range(len(epoch_list*2))[::2]
