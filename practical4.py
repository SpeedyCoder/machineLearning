import tensorflow as tf
import numpy as np

from time import time
import reader

# Define constants and load data
learning_rate = 0.001
max_grad_norm = 5

epochs = 10
batch_size = 1

MAX_SIZE = 7020
vocab, E_talks, E_keywords, talks_dict, keywords_dict = reader.get_generation_data(1585, 250, 250)
batches_train = reader.make_batches_gen(talks_dict["train"], keywords_dict["train"], batch_size)

vocab_size = E_talks.shape[0]
embedding_size = E_talks.shape[1]

keys_size = E_keywords.shape[1]

size = 50

# tf Graph input
inputs = tf.placeholder("int32", [None, None])
targets = tf.placeholder("int32", [None, None])
seq_lengths = tf.placeholder("int32", [None])
loss_weights = tf.placeholder("float", [None, None])


keywords = tf.placeholder("int32", [None, None])
keep_prob = tf.placeholder(tf.float32)

# Define weights
weights = {
    'W': tf.Variable(tf.random_normal([size, vocab_size])),
    'gate': tf.Variable(tf.random_normal([1, embedding_size, keys_size]))
}
biases = {
    'b': tf.Variable(tf.random_normal([vocab_size])),
    'gate': tf.Variable(tf.random_normal([keys_size]))
}


def RNN(inputs, keywords, seq_lengths, E_talks, E_keywords, weights, biases):
    x = tf.nn.embedding_lookup(E_talks, inputs)
    x = tf.transpose(x, [1, 0, 2])

    x2 = tf.nn.embedding_lookup(E_keywords, keywords)
    x2 = tf.reduce_mean(x2, 1)

    x2 = tf.reshape(x2, [-1, 1, 20])
    x2 = tf.tile(x2, [1,tf.shape(x)[0] ,1])
    x2 = tf.transpose(x2, [1, 0, 2])

    w = tf.tile(weights['gate'], [tf.shape(x)[0], 1, 1])

    s = tf.sigmoid(tf.matmul(x, w))
    x2 = s*x2

    x = tf.concat([x, x2], 2)
    

    cell = tf.contrib.rnn.GRUCell(size)

    outputs, state = tf.nn.dynamic_rnn(cell, x, time_major=True, 
        dtype=tf.float32, sequence_length=seq_lengths)

    output = tf.reshape(outputs, [-1, size])

    return outputs, tf.matmul(output, weights['W']) + biases['b']

outputs, logits = RNN(inputs, keywords, seq_lengths, E_talks, E_keywords, weights, biases)

pred = tf.argmax(logits, 1)

loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
    [logits],
    [tf.reshape(targets, [-1])],
    [loss_weights])

cost = tf.reduce_sum(loss) / tf.cast(tf.reduce_sum(seq_lengths), tf.float32)

# Gradient clipping
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.apply_gradients(
    zip(grads, tvars),
    global_step=tf.contrib.framework.get_or_create_global_step())

init = tf.global_variables_initializer()

def sample(sess):
    ins = [[2]]
    res = []
    length = 1
    word_index = sess.run(pred, feed_dict={
        inputs: ins,
        keywords: batches_train[0]["keywords"],
        seq_lengths: [length]
    })[0]
    res.append(vocab[word_index])
    ins[0].append(word_index)
    length += 1

    # while word_index != 4:
    while length < 50:
        word_index = sess.run(pred, feed_dict={
            inputs: ins,
            keywords: batches_train[0]["keywords"],
            seq_lengths: [length]
        }) [length - 1]
        
        res.append(vocab[word_index])
        ins[0].append(word_index)
        length += 1

    print(' '.join(res))

def calculate_perplexity(step, sess):
    res = 0
    for batch in batches_train:
        res += sess.run([cost], feed_dict={
            inputs: batch["inputs"],
            keywords: batch["keywords"],
            targets: batch["targets"],
            seq_lengths: batch["seq_lengths"],
            loss_weights: batch["loss_weights"]})

    res = res/len(batches_train)
    print("%s cost: %.3f, perplexity: %.3f" %
                (step, res, np.exp(-res)))


sess = tf.Session()
sess.run(init)

start = time()

for step in range(epochs):
    # Calculate batch accuracy
    for i, batch in enumerate(batches_train):
        # Run optimization op (backprop)
        # pprint(batch)
        # TODO: check if using seq lengths works or need to use a mask
        # outs = sess.run(outputs, feed_dict={
        #     inputs: batch["inputs"], 
        #     targets: batch["targets"],
        #     seq_lengths: batch["seq_lengths"],
        #     loss_weights: batch["loss_weights"]})

        # print(outs[-1])

        costs, _ = sess.run([cost, train_op], feed_dict={
            inputs: batch["inputs"],
            keywords: batch["keywords"],
            targets: batch["targets"],
            seq_lengths: batch["seq_lengths"],
            loss_weights: batch["loss_weights"]})

        print("Batch %s done, cost: %.3f, time: %.3fs" %
                (i, costs, time() - start))

    print("Time taken: %.3f" % (time() - start))
    calculate_perplexity(step, sess)
    sample(sess)

print("Finished in: %.3f"% (time() - start))






