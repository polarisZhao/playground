# -*- coding:utf-8 -*-
import tensorflow as tf
import utils
import os
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs
        param image_width: The input image width
        param image_height: The input image height
        param image_channels: The number of image channels
        param z_dim: The dimension of Z
        return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """
    input_ph = tf.placeholder(shape = [None, image_width, image_height, image_channels], dtype=tf.float32)
    z_ph = tf.placeholder(shape = [None, z_dim], dtype=tf.float32)
    lr_ph = tf.placeholder(dtype=tf.float32)
    return input_ph, z_ph, lr_ph


def discriminator(images, reuse=False, is_train=True,):
    """
    Create the discriminator network
        param image: Tensor of input image(s)
        param reuse: Boolean if the weights should be reused
        return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """
    df_dim = 128
    alpha = 0.2
    with tf.variable_scope("discriminator", reuse=reuse):
        # conv1 images -> 128 * 32 * 32
        conv1 = tf.layers.conv2d(images, df_dim, 5, strides=2, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='same')
        bn1 = tf.layers.batch_normalization(conv1, training=is_train)
        relu1 = tf.maximum(alpha*bn1, bn1)
        x1 = tf.layers.dropout(relu1, rate=0.6)

        # conv2 128 * 32 * 32 -> 256 * 16 * 16
        conv2 = tf.layers.conv2d(x1, df_dim*2, 5, strides = 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='same')
        bn2 = tf.layers.batch_normalization(conv2, training=is_train)
        relu2 = tf.maximum(alpha * bn2, bn2)
        x2 = tf.layers.dropout(relu2, rate=0.6)

        # conv3 256 * 16 * 16 ->  512 * 8 * 8
        conv3 = tf.layers.conv2d(x2, df_dim*4, 5, strides = 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='same')
        bn3 = tf.layers.batch_normalization(conv3, training=is_train)
        relu3 = tf.maximum(alpha*bn3, bn3)
        x3 = tf.layers.dropout(relu3, rate=0.6)

        # conv4 512 * 8 * 8 -> 1024 * 4 * 4
        conv4 = tf.layers.conv2d(x3, df_dim*8, 5, strides = 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='same')
        bn4 = tf.layers.batch_normalization(conv4, training=is_train)
        relu4 = tf.maximum(alpha*bn4, bn4)
        x4 = tf.layers.dropout(relu4, rate=0.6)

        # flatten
        flatten = tf.reshape(x4, (-1, 4*4*df_dim*8))


        # fully connection
        logits = tf.layers.dense(flatten, 1, activation=None)
        out = tf.sigmoid(logits)

        return out, logits


def generator(z, out_channel_dim, is_train=True):
    """
    Create the generator network
        param z: Input z
        param out_channel_dim: The number of channels in the output image
        param is_train: Boolean if generator is being used for training
        return: The tensor output of the generator
    """
    alpha = 0.2
    gf_dim = 128
    with tf.variable_scope('generator', reuse=not is_train):
        # fully connection & Reshape
        h1 = tf.layers.dense(z, 4*4*1024, activation=None)

        # Reshape
        h1 = tf.reshape(h1, (-1, 4, 4, gf_dim*8))
        bn1 = tf.layers.batch_normalization(h1, training=is_train)
        relu1 = tf.maximum(alpha * bn1, bn1)
        x1 = tf.layers.dropout(relu1, rate=0.6)

        # conv2   4 * 4 * 1024 -> 8 * 8 * 512
        conv2 = tf.layers.conv2d_transpose(x1, gf_dim*4, 5, strides=2,kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='same', activation=None)
        bn2 = tf.layers.batch_normalization(conv2, training=is_train)
        relu2 = tf.maximum(alpha * bn2, bn2)
        x2 = tf.layers.dropout(relu2, rate=0.6)

        # conv3  8 * 8 * 512 -> 16 * 16 * 256
        conv3 = tf.layers.conv2d_transpose(x2, gf_dim*2, 5, strides=2, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='same', activation=None)
        bn3 = tf.layers.batch_normalization(conv3, training=is_train)
        relu3 = tf.maximum(alpha * bn3, bn3)
        x3 = tf.layers.dropout(relu3, rate=0.6)

        # conv4  16 * 16 * 256 -> 32 * 32 * 128
        conv4 = tf.layers.conv2d_transpose(x3, gf_dim, 5, strides=2, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='same', activation=None)
        bn4 = tf.layers.batch_normalization(conv4, training=is_train)
        relu4 = tf.maximum(alpha * bn4, bn4)
        x4 = tf.layers.dropout(relu4, rate=0.6)

        # conv5 32 * 32 * 128 -> 64 * 64 * 3
        logits = tf.layers.conv2d_transpose(x4, out_channel_dim, 5, strides=2,kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='same', activation=None)
        out = tf.tanh(logits)

    return out

def model_loss(input_real, input_z, out_channel_dim):
    """
    Get the loss for the discriminator and generator
        param input_real: Images from the real dataset
        param input_z: Z input
        param out_channel_dim: The number of channels in the output image
        return: A tuple of (discriminator loss, generator loss)
    """
    # Build network
    g_model = generator(input_z, out_channel_dim)

    d_model_real, d_logits_real = discriminator(input_real)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)

    # Calculate losses
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                         labels= 0.9 * tf.ones_like(d_logits_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                        labels=tf.zeros_like(d_logits_real)))
    d_loss = d_loss_real + d_loss_fake
    d_loss = tf.identity(d_loss)

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                    labels=tf.ones_like(d_logits_fake)))
    return d_loss, g_loss


def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
        param d_loss: Discriminator loss Tensor
        param g_loss: Generator loss Tensor
        param learning_rate: Learning Rate Placeholder
        param beta1: The exponential decay rate for the 1st moment in the optimizer
        return: A tuple of (discriminator training operation, generator training operation)
    """
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt


def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    Train the GAN
        param epoch_count: Number of epochs
        param batch_size: Batch Size
        param z_dim: Z dimension
        param learning_rate: Learning Rate
        param beta1: The exponential decay rate for the 1st moment in the optimizer
        param get_batches: Function to get batches
        param data_shape: Shape of the data
        param data_image_mode: The image mode to use for images ("RGB" or "L")
    """
    _, width, height, channel = data_shape

    input_, z_, lr_ = model_inputs(width, height, channel, z_dim)
    d_loss, g_loss = model_loss(input_, z_, channel)
    d_train_opt, g_train_opt = model_opt(d_loss, g_loss, lr_, beta1)

    steps = 0
    checkpoint_interval = 1024
    saver = tf.train.Saver()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            for batch_images in get_batches(batch_size):
                steps += 1
                batch_images = batch_images * 2

                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))

                _ = sess.run(d_train_opt, feed_dict={input_: batch_images, z_: batch_z, lr_:learning_rate})
                _ = sess.run(g_train_opt, feed_dict={input_: batch_images, z_: batch_z, lr_:learning_rate})

                if steps % 20 == 0:
                    train_loss_d = d_loss.eval({z_: batch_z, input_: batch_images})
                    train_loss_g = g_loss.eval({z_: batch_z})

                    print("Epoch {}/{}...".format(epoch_i+1, epoch_count),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))

                if steps % checkpoint_interval == 0:
                    n_images = 49
                    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])
                    samples = sess.run(generator(z_, channel, False),
                              feed_dict={z_: example_z})
                    images_grid = utils.images_square_grid(samples, data_image_mode)
                    utils.show_generator_output(samples, images_grid, data_image_mode, steps)
                    saver.save(sess, './checkpoint/dcgan-model', global_step=steps) # checkpoint



def main():
    epochs = 10
    learning_rate = 0.0002
    beta1 = 0.5

    batch_size = 64
    z_dim = 100

    celeba_dataset = utils.Dataset('celeba',
                                     glob(os.path.join(data_dir, 'img_align_celeba/*.jpg'))) # read datasets
    with tf.Graph().as_default():
         train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
                       celeba_dataset.shape, celeba_dataset.image_mode)


if __name__ == "__main__":
    main()
