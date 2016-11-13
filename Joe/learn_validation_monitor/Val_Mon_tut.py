#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 00:52:08 2016

@author: yokian
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

# Data sets
IRIS_TRAINING = "iris.csv"
IRIS_TEST = "iris_test.csv"
iris=tf.contrib.learn.python.learn.datasets.load_iris()
training_data=np.float32(np.vstack((iris.data[::3],iris.data[1::3])))
test_data=np.float32(iris.data[2::3])
class dset(object):
    data=0
    target=0
    def __init__(self,data,target):
        self.data = data
        self.target = target
        
training_set = dset(training_data,(np.hstack((iris.target[::3],iris.target[1::3]))))
test_set = dset(test_data,(iris.target[2::3]))
    
# Load datasets.
#training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#    filename=IRIS_TRAINING, target_dtype=np.int, features_dtype=np.float32)
#test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#    filename=IRIS_TEST, target_dtype=np.int, features_dtype=np.float32)

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
                   
validation_metrics = {"accuracy": tf.contrib.metrics.streaming_accuracy,
                      "precision": tf.contrib.metrics.streaming_precision,
                      "recall": tf.contrib.metrics.streaming_recall}
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50,
    metrics=validation_metrics)
# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")

# Fit model.
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000,monitors=[validation_monitor])

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new flower samples.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = classifier.predict(new_samples)
print('Predictions: {}'.format(str(y)))