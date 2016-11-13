
Ylabel_train = train_data[:,label_col_index]
Ylabel_test = test_data[:,label_col_index]
Y_train = train_data[:,output_index_start:]
Y_test = test_data[:,output_index_start:]
X_train = train_data[:,:label_col_index] #-1 to skip last column
X_test = test_data[:,:label_col_index]

X_train = np.reshape(X_train,(n_train,32,32,3))  
