# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import input_data

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main():
    # read the datasets
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print('Training set', mnist.train.images.shape, mnist.train.labels.shape)
    print('Validation set', mnist.validation.images.shape, mnist.validation.labels.shape)
    print('Test set', mnist.test.images.shape, mnist.test.labels.shape)

    # Build the Graphs
    x = tf.placeholder("float", [None, 784])  # 占位符, 输入
    y_ =  tf.placeholder("float", [None,10])   #占位符，输出

    x_image = tf.reshape(x, [-1,28,28,1])
    #    tf.summary.image('input', x_image, 5)
    W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))  #truncated_normal 正太分布 32 个 (5*5)，标准差为0.1
    b_conv1 = tf.Variable(tf.constant(0.1, shape = [32]))
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second conv, relu, max_pool
    W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(0.1, shape = [64]))
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    # fc layers, relu
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape = [1024]))
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout layers
    keep_prob = tf.placeholder("float")       # 将keep_prob 占位，运行的时候传入
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # full connection layers,  softmax
    W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape = [10]))
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # cross_entropy
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))   # 损失函数是目标类别和预测类别之间的交叉熵,将每一副图像求和
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step=global_step) 
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # cast将bool类型 转化为 0，1

    with tf.name_scope("summaries"):    # Step One
        tf.summary.image('input', x_image, 5)
        tf.summary.scalar("loss", cross_entropy)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.histogram("loss", cross_entropy)
        summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()   # Step Two

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./graphs", sess.graph)  # Step Two

        ckpt = tf.train.get_checkpoint_state(os.path.dirname('./checkpoints/ckpt'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)


        for i in range(1000):
            batch = mnist.train.next_batch(50)

            # per on the eval data size
            if i % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_:batch[1], keep_prob: 1.0})
                print("step: {0:}, training accuracy {1:}".format(i, train_accuracy))

            # checkpoint 
            if (i + 1)% 500 == 0:
                saver.save(sess, './checkpoints/ckpt', global_step=global_step)    # Step Three

#            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            summary = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  # Step Three

            writer.add_summary(summary, global_step = i)  # Step Four

        # Runing on the Test data
        print("test accuracy {0:}".format(accuracy.eval(feed_dict={
            x: mnist.test.images, y_:mnist.test.labels, keep_prob: 1.0})))   #测试的时候不随抛弃


if __name__ == "__main__":
    main()
