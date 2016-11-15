

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
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))#+\
#                        beta[1]*tf.nn.l2_loss(weights['h2'])+\
#                        beta[2]*tf.nn.l2_loss(weights['h3']) +beta[3]*tf.nn.l2_loss(weights['out']) )
#    beta*tf.nn.l2_loss(biases['b1']) +\
#    beta*tf.nn.l2_loss(weights['out']) +\
#    beta*tf.nn.l2_loss(biases['out']))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
if restore_save== False:
    init = tf.initialize_all_variables()

saver = tf.train.Saver()
epoch_list=[]
accuracy_list=[]

training_epochs = 100
batch_size = 100
display_step = 2
n_hidden_1 = 50
n_hidden_2 = 20
beta = 1e-5 

#now run this separately

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
