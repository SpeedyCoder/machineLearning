import tensorflow as tf
from tensorflow.contrib import rnn

import reader


# Parameters
learning_rate = 0.001
epochs = 50
batch_size = 30
display_step = 1

# Network Parameters
n_input = 50
# n_steps = X.shape[1] # timesteps
n_hidden = 128 # hidden layer num of features
n_size = 50
n_classes = 8 # total classes


# Read and process the data
E, X_dict, y_dict = reader.get_raw_data(1585, 250)
batches = reader.make_batches(X_dict["train"], y_dict["train"], batch_size)


# tf Graph input
x = tf.placeholder("int32", [None, None])
y = tf.placeholder("float", [None, n_classes])
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


def RNN(x, embedding, weights, biases):
#     # Get embeddings for the talk
    x = tf.nn.embedding_lookup(embedding, x)
#     print(tf.shape(x))
    
#     x = tf.unstack(x, num=n_steps, axis=1)
#     print(tf.shape(x))

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    print("Creating the network")
    
#     # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
#     # Reshaping to (n_steps*batch_size, n_input)
#     x = tf.reshape(x, [-1, n_input])
#     # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
#     x = tf.split(x, n_steps, 0)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    print("Creating the static rnn")
    # Get lstm cell output
    # outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, time_major=True, dtype=tf.float32)
    print(tf.shape(outputs))
    
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


# In[ ]:

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        for batch_x, batch_y in batches:
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
        if i % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
            # Calculate batch loss
            # loss = sess.run(cost, feed_dict={x: X_train, y: Y_train, keep_prob: 1})
            print ("Epochs:", i, ",", "Training Accuracy=", "{:.5f}".format(acc))

    print ("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
#     print ("Testing Accuracy:",
#             sess.run(accuracy, feed_dict={x: X_test, y: Y_test, keep_prob: 1}))


# In[8]:

tf.reset_default_graph()


# In[ ]:



