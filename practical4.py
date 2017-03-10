import tensorflow as tf
import numpy as np
from scipy import stats

from time import time

# TODO:
# - add attention
# - make predictions faster
    
class Config(object):
    """docstring for Config"""
    def __init__(self, data):
        self.learning_rate = 1.0
        self.lr_decay = 1 / 1.15

        self.max_grad_norm = 10

        self.epochs = 10
        self.batch_size = 1
        
        self.vocab_size = len(data.vocab)
        print("Size of vocabulary:", self.vocab_size)
        self.embedding_size = data.E_talks.shape[1]
        self.keys_size = data.E_keywords.shape[1]

        self.rnn_size = 200

        self.step_sample = 20
        

class Model(object):
    def __init__(self, data, config):
        self.data = data
        self.config = config

        tf.reset_default_graph()
        print("Building the RNN...")

        # tf Graph input
        self.inputs = tf.placeholder("int32", [None, None])
        self.targets = tf.placeholder("int32", [None, None])
        self.keywords = tf.placeholder("int32", [None, None])
        self.keep_prob = tf.placeholder(tf.float32)

        self.seq_lengths = tf.placeholder("int32", [None])
        self.loss_weights = tf.placeholder("float", [None, None])
        # self.mask = tf.placeholder("float", [None, None])

        # Learning rate
        self.lr = tf.Variable(0.0, trainable=False)
        self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self.lr_update = tf.assign(self.lr, self.new_lr)

        # For generation
        self.state_in = tf.placeholder("float", [None, config.rnn_size])

        # Define weights
        self.weights = {
            'W': tf.Variable(tf.random_normal([config.rnn_size, config.vocab_size])),
            'gate': tf.Variable(tf.random_normal([1, config.embedding_size, config.keys_size]))
        }
        self.biases = {
            'b': tf.Variable(tf.random_normal([config.vocab_size])),
            'gate': tf.Variable(tf.random_normal([1, config.keys_size]))
        }

        x = tf.nn.embedding_lookup(data.E_talks, self.inputs)
        x2 = tf.nn.embedding_lookup(data.E_keywords, self.keywords)

        x = tf.transpose(x, [1, 0, 2])
        x2 = tf.reduce_mean(x2, 1)

        # x2 = tf.reshape(x2, [-1, 1, config.keys_size])
        # x2 = tf.tile(x2, [1,tf.shape(x)[0] ,1])
        # x2 = tf.transpose(x2, [1, 0, 2])
        x2 = tf.reshape(x2, [1, -1, config.keys_size])
        x2 = tf.tile(x2, [tf.shape(x)[0], 1, 1])

        w = tf.tile(self.weights['gate'], [tf.shape(x)[0], 1, 1])
        b = tf.tile(self.biases['gate'], [tf.shape(x)[0], 1])

        s = tf.sigmoid(tf.matmul(x, w))
        x2 = s*x2

        x = tf.concat([x, x2], 2)
        self.x = tf.nn.dropout(x, self.keep_prob)

        self.cell = tf.contrib.rnn.GRUCell(config.rnn_size)

        self.outputs, self.state = tf.nn.dynamic_rnn(self.cell, self.x, time_major=True, 
            dtype=tf.float32, sequence_length=self.seq_lengths, scope="rnn")

        output = tf.reshape(self.outputs, [-1, config.rnn_size])
        output = tf.nn.dropout(output, self.keep_prob)

        self.logits = tf.matmul(output, self.weights['W']) + self.biases['b']

        with tf.variable_scope("rnn"):
            tf.get_variable_scope().reuse_variables()
            single_out, self.single_state = self.cell(self.x[-1], self.state_in)
        logs = tf.matmul(single_out, self.weights['W']) + self.biases['b']
        self.single_probs = tf.nn.softmax(logs)

        # pred = tf.argmax(logits, 1)
        self.probs = tf.nn.softmax(self.logits[-1])

        self.loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [self.logits],
            [tf.reshape(self.targets, [-1])],
            [self.loss_weights],
            average_across_timesteps=False)

        cost = tf.reduce_sum(self.loss) / tf.cast(tf.reduce_sum(self.seq_lengths), tf.float32)
        self.cost = cost / tf.cast(tf.shape(self.inputs)[0], tf.float32)

        # Gradient clipping
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), config.max_grad_norm)

        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self.init = tf.global_variables_initializer()

    def sample(self, sess, limit=100):
        keys = ["talks", "entertainment", "performance", "music"]
        keys = [[self.data.keys_index_map[key] for key in keys]]

        indexes = np.arange(0, self.config.vocab_size)
        ins = [[2]]
        res = []
        length = 1
        word_index = 0

        state, prob = sess.run([self.state ,self.probs], feed_dict={
            self.inputs: ins,
            self.keywords: keys,
            self.seq_lengths: [length],
            self.keep_prob: 1.0
        })

        prob = prob/prob.sum()

        dist = stats.rv_discrete(name='custm', values=(indexes, prob))
        for i in range(5):
            word_index = dist.rvs()
            if word_index != 4:
                break
        res.append(self.data.vocab[word_index])
        ins[0].append(word_index)
        length += 1

        # while word_index != 4:
        while length < limit and word_index != 4:
            state, prob = sess.run([self.single_state, self.single_probs], 
                feed_dict={
                    self.inputs: ins,
                    self.keywords: keys,
                    self.state_in: state,
                    self.keep_prob: 1.0
                })

            prob = prob/prob.sum()

            dist = stats.rv_discrete(name='custm', values=(indexes, prob.T))
            word_index = dist.rvs()
            res.append(self.data.vocab[word_index])
            ins[0].append(word_index)
            length += 1

        return (' '.join(res))

    def calculate_perplexity(self, step, sess):
        res = 0
        for batch in self.data.batches_validate:
            res += sess.run(self.cost, feed_dict={
                self.inputs: batch["inputs"],
                self.keywords: batch["keywords"],
                self.targets: batch["targets"],
                self.seq_lengths: batch["seq_lengths"],
                self.loss_weights: batch["loss_weights"],
                self.keep_prob: 1.0})

        res = res/len(self.data.batches_train)

        return res

    def train(self):
        self.saver = tf.train.Saver()
        print("Training...")
        learning_rate = self.config.learning_rate
        step_size = self.config.step_sample
        start = time()

        sess = tf.Session()
        sess.run(self.init)
        sess.run(self.lr_update, feed_dict={self.new_lr: learning_rate})
        # self.saver.restore(sess, "./model.ckpt")
        # print(self.sample(sess, limit=100))
        start = time()

        for step in range(self.config.epochs):

            # Calculate batch accuracy
            start_epoch = time()
            start_batch = start_epoch
            costs = 0
            for i, batch in enumerate(self.data.batches_train):
                # Run optimization op (backprop)
                # pprint(batch)
                # TODO: check if using seq lengths works or need to use a mask
                # outs = sess.run(outputs, feed_dict={
                #     inputs: batch["inputs"], 
                #     targets: batch["targets"],
                #     seq_lengths: batch["seq_lengths"],
                #     loss_weights: batch["loss_weights"]})

                # print(outs[-1])
                cost, _ = sess.run([self.cost, self.train_op], feed_dict={
                    self.inputs: batch["inputs"],
                    self.keywords: batch["keywords"],
                    self.targets: batch["targets"],
                    self.seq_lengths: batch["seq_lengths"],
                    self.loss_weights: batch["loss_weights"],
                    self.keep_prob: 0.5})

                costs += cost


                if (i+1) % step_size == 0 and i != 0:
                    t = time() - start_batch
                    print("%s talks done, avg cost: %.2f, time: %.2fs, avg time for talk: %.3fs" %
                            (i+1, costs/step_size, t, t/step_size))

                    start_batch = time()
                    costs = 0

                if (i+1) % (step_size*10) == 0 and i != 0:
                    print(self.sample(sess, limit=100))

            end_epoch = time()
            print("-"*70)
            # cost = self.calculate_perplexity(step, sess)
            print("Epoch %s complete, validation cost: %.2f, time: %.2fs, avg time for talk: %.3fs\n" % 
                    (step+1, cost, time() - start_epoch, (end_epoch - start_epoch)/len(self.data.batches_train)))

            talk = self.sample(sess, limit=1000)
            print(talk, "\n")
            with open('talk'+str(step)+'.txt', 'w') as f:
                f.write(talk)

            # Decrease learning rate
            learning_rate = learning_rate * self.config.lr_decay
            sess.run(self.lr_update, feed_dict={self.new_lr: learning_rate})

        print("Finished in: %.3f"% (time() - start))
        self.saver.save(sess, "model.ckpt")
        sess.close()





