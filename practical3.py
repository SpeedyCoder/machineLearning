import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import math

import reader
from balancer import Balancer
from collections import Counter

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
n_size = 30
n_classes = 8 # total classes

# reg_weights = tf.constant(np.zeros(8, dtype=np.float32))
# y = tf.placeholder("float", [batch_size, n_classes])
# x = tf.placeholder("float", [batch_size, n_classes])
# z = tf.placeholder("float", [3])
# weight = tf.reduce_sum(reg_weights*y, axis=1)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)*weight)

# exit()


# Read and process the data
E, X_dict, y_dict = reader.get_raw_data(1550, 250, 250, MAX_SIZE=MAX_SIZE)
print(len(X_dict["train"]), len(X_dict["train"][0]))
print(len(y_dict["train"]))
reg_weights = np.zeros(8, dtype=np.float32)
for c in range(8):
    reg_weights[c] = (y_dict["train"][:,c] == 1).sum()

validation_counts = [0 for _ in range(8)]
for c in range(8):
    validation_counts[c] = int((y_dict["validate"][:,c] == 1).sum())
print(validation_counts)
 
print(reg_weights)

reg_weights = 1 + np.log(reg_weights.max()) - np.log(reg_weights)

print(reg_weights)
reg_weights = tf.constant(reg_weights.T)

balancer = Balancer(X_dict["train"], y_dict["train"])
batches_train = reader.make_batches(X_dict["train"], y_dict["train"], batch_size)
batches_validate = reader.make_batches(X_dict["validate"], y_dict["validate"], batch_size)
batches_test = reader.make_batches(X_dict["test"], y_dict["test"], batch_size)

del (X_dict, y_dict)
print("Data processed.")


# tf Graph input
x = tf.placeholder("int32", [batch_size, None])
y = tf.placeholder("float", [batch_size, n_classes])
keep_prob = tf.placeholder(tf.float32)

# Define weights
weights = {
    'W': tf.Variable(tf.random_normal([n_hidden, n_size])),
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

    parts = math.ceil(MAX_SIZE/n_steps)
    print("Parts:", parts)
    # splits = tf.split(x, math.ceil(MAX_SIZE/n_steps), axis=0)

    # Define a lstm cell with tensorflow
    # cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=False)
    # cell = rnn.BasicRNNCell(n_hidden)
    cell = rnn.GRUCell(n_hidden)
    state = cell.zero_state(batch_size, tf.float32)
    print(tf.shape(state))

    print("Creating the truncated rnn")
    outputs = []
    result = {}
    with tf.variable_scope("TBPTT"):
        for i in range(parts):
            if i > 0: tf.get_variable_scope().reuse_variables()
            result[i], state = rnn.static_rnn(cell, inputs[n_steps*i:n_steps*(i+1)], initial_state=state)
            # result[i], state = tf.nn.dynamic_rnn(cell, x[:,n_steps*i:n_steps*(i+1),:], time_major=True, 
            #                                   dtype=tf.float32, initial_state=state)
            print("here")
            state = tf.stop_gradient(state)
            outputs.append(result[i])

    outputs = tf.concat(outputs, 0)
    z = outputs[-1]
    # z = tf.reduce_mean(outputs, 0)
    # z_drop = tf.nn.dropout(z, keep_prob)

    h = tf.nn.tanh(tf.matmul(z, weights['W']) + biases['b'])
    
    # Dropout layer
    h_drop = tf.nn.dropout(h, keep_prob)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(h_drop, weights['V']) + biases['c']

pred = RNN(x, E, weights, biases)
print("Created the network")
# Define loss and optimizer
# weight = tf.reduce_sum(reg_weights*y, axis=1)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)*weight)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# reg = tf.reduce_sum((pred*y)*reg_weights)
# cost = tf.reduce_mean(-tf.reduce_sum(reg_weights*y * tf.log(pred), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.AdadeltaOptimizer(learning_rate=1).minimize(cost)

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




