import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import math

import reader
from balancer import Balancer

from time import time

start = time()

# Parameters
learning_rate = 0.001
epochs = 100
batch_size = 50
display_step = 1
MAX_SIZE = 1000

# Network Parameters
n_input = 50
n_steps = 100
n_hidden = 50 # hidden layer num of features
n_size = 400
n_classes = 8 # total classes


# Read and process the data
E, X_dict, y_dict = reader.get_raw_data(1500, 250, 250, MAX_SIZE=MAX_SIZE)

balancer = Balancer(X_dict["train"], y_dict["train"])
batches_train = reader.make_batches(X_dict["train"], y_dict["train"], batch_size)
batches_validate = reader.make_batches(X_dict["validate"], y_dict["validate"], batch_size)
batches_test = reader.make_batches(X_dict["test"], y_dict["test"], batch_size)

del (X_dict, y_dict)
print("Data processed.")


# tf Graph input
x = tf.placeholder("int32", [None, None])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# Define weights
weights = {
    'W': tf.Variable(tf.random_normal([2*n_hidden, n_size])),
    'V': tf.Variable(tf.random_normal([n_size, n_classes]))
}
biases = {
    'b': tf.Variable(tf.random_normal([n_size])),
    'c': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(inp, embedding, weights, biases):
    # Get embeddings for the talk
    x = tf.nn.embedding_lookup(embedding, inp)
    # x = tf.nn.dropout(x, keep_prob)

    print("Creating the network")
    # Permuting batch_size and n_steps
    # x = tf.transpose(x, [1, 0, 2])
    # inputs = tf.unstack(x, num=MAX_SIZE, axis=1)

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    inputs = tf.split(x, MAX_SIZE, 0)

    # splits = tf.split(x, math.ceil(MAX_SIZE/n_steps), axis=0)

    # Define a lstm cell with tensorflow
    # fw_cell = rnn.GRUCell(n_hidden)
    # bw_cell = rnn.GRUCell(n_hidden)
    cell = rnn.GRUCell(n_hidden)

    print("Creating the bidirectional rnn")
    #  outputs, _, _ = rnn.static_bidirectional_rnn(fw_cell, bw_cell, inputs, dtype=tf.float32)
    outputs, _, _ = rnn.static_bidirectional_rnn(cell, cell, inputs, dtype=tf.float32)

    z = tf.reduce_mean(outputs, 0)

    h = tf.nn.tanh(tf.matmul(z, weights['W']) + biases['b'])
    
    # Dropout layer
    h_drop = tf.nn.dropout(h, keep_prob)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(h_drop, weights['V']) + biases['c']

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
print("Finished creating the model.")

def get_accuracy(batches):
    acc = 0
    for batch_x, batch_y in batches:
        acc += sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})

    return acc/len(batches)



accs_train = []
accs_validate = []
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # batch = batches_train[0]
    # print(sess.run(weight, feed_dict={y: batch[1]}))
    print ("Testing Accuracy:", get_accuracy(batches_test))
    for i in range(epochs):
        # Calculate batch accuracy
        acc_train = get_accuracy(batches_train)
        acc_validate = get_accuracy(batches_validate)
        print ("Epoch:", i, ",", "Training Accuracy=", "{:.5f}".format(acc_train),
               "Validation Accuracy=", "{:.5f}".format(acc_validate))
        accs_train.append(acc_train)
        accs_validate.append(acc_validate)

        for _ in range(31):
            batch_x, batch_y = balancer.next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            

    print ("Optimization Finished!")

    print ("Testing Accuracy:", get_accuracy(batches_test))

np.save("accs_train", accs_train)
np.save("accs_validate", accs_validate)

end = time()
print("Finished in:", end - start, "seconds")




