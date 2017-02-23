
# coding: utf-8

# '''
# A Reccurent Neural Network (LSTM) implementation example using TensorFlow library.
# This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
# Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
# 
# Author: Aymeric Damien
# Project: https://github.com/aymericdamien/TensorFlow-Examples/
# '''

# In[1]:

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

print("Loading data...")
X = np.load("talks.npy")
print("Shape of X:", X.shape)
Y = np.load("keywords.npy")
print("Shape of y:", Y.shape)
E = np.load("embedding.npy")
print("Shape of E:", Y.shape)

Y_train = Y[:1585]
X_train = X[:1585]

Y_validation = Y[1585:1835]
X_validation = X[1585:1835]

Y_test = Y[1835:]
X_test = X[1835:]


# '''
# To classify images using a reccurent neural network, we consider every image
# row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
# handle 28 sequences of 28 steps for every sample.
# '''

# In[3]:

# Parameters
learning_rate = 0.001
epochs = 10
batch_size = 50
display_step = 10

# Network Parameters
n_input = 50
n_steps = X.shape[1] # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = Y.shape[1] # total classes

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# In[ ]:

def RNN(x, embedding, weights, biases):
#     # Get embeddings for the talk
#     x = tf.nn.embedding_lookup(embedding, x)
#     print(tf.shape(x))
    
#     x = tf.unstack(x, num=n_steps, axis=1)
#     print(tf.shape(x))

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    print("Creating the network")
    
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    print("Creating the static rnn")
    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, E, weights, biases)
print("Created the network")
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


# In[6]:

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        for j in range(32):
            batch_x = X_train[(j%batch_size)+i*batch_size: (j%batch_size)+(i+1)*batch_size]
            batch_y = Y_train[(j%batch_size)+i*batch_size: (j%batch_size)+(i+1)*batch_size]
            
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print ("Iter", step*batch_size, ", Minibatch Loss=",
                      "{:.6f}".format(loss), ", Training Accuracy=", "{:.5f}".format(acc))

    print ("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print ("Testing Accuracy:",
            sess.run(accuracy, feed_dict={x: test_data, y: test_label}))


# In[ ]:



