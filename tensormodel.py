import os
import sys
import tensorflow as tf
import pickle
data_set = pickle.load(open('data/reduced_data.bin', 'rb'))

for i in range(1000):
#ls = list(input_)
    #print data_set.shape
    #t = tf.stack()

    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
#W = tf.Variable(tf.zeros([1024, 10]))
#b = tf.Variable(tf.zeros([10]))
#init = tf.global_variables_initializer()
#Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 1024]), W) + b)
    W = tf.Variable(tf.zeros([1024,10]))
    b = tf.Variable(tf.zeros([10]))
    init = tf.initialize_all_variables()

    #model
    Y = tf.nn.softmax(tf.matmul(tf.reshape(X,[-1,1024]),W) + b)
#place holder for correct lables
    Y_ = tf.placeholder(tf.float32, [None, 10])
#print Y_
# loss function
    cross_entropy = -tf.reduce_sum(Y * tf.log(Y_))
#print cross_entropy
# % of correct answers found in batch
    is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
#print accuracy
    optimizer = tf.train.GradientDescentOptimizer(0.003)
    train_step = optimizer.minimize(cross_entropy)

    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_X, batch_Y =
#     train_data = tensor_

     # train
#     sess.run(train_step, feed_dict = train_data)
#     a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

     # success on test data
