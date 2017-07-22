import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_DATA/', one_hot=True)

batch_size = 128
num_classes = 10
epochs = 12
img_rows, img_cols = 28, 28


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[
                       None, img_rows * img_cols], name='x_in')
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_in')
with tf.name_scope('conv_param'):
    conv_w = weight_variable([3, 3, 1, 32], name='conv_w')
    conv_b = bias_variable([32], name='conv_b')
with tf.name_scope('fc_param'):
    fc_w = weight_variable([13 * 13 * 32, 10], name='fc_w')
    fc_b = bias_variable([10], name='fc_b')

x_image = tf.reshape(x, [-1, img_rows, img_cols, 1])
conv_out = tf.nn.relu(conv2d(x_image, conv_w) + conv_b)
pool_out = max_pool_2x2(conv_out)
pool_flat = tf.reshape(pool_out, [-1, 13 * 13 * 32])
fc_out = tf.matmul(pool_flat, fc_w) + fc_b

with tf.name_scope('cost'):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=fc_out))
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
with tf.name_scope('accurcy'):
    correct_pred = tf.equal(tf.argmax(fc_out, axis=1), tf.argmax(y_, axis=1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

n_iteration = mnist.train.num_examples / batch_size * epochs
with tf.Session() as sess:
    writer = tf.summary.FileWriter('tf_output', graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    writer.close()
    import time
    beg = time.time()
    for i in range(n_iteration):
        batch = mnist.train.next_batch(batch_size)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        if i % 100 == 0:
            train_loss, train_acc = sess.run(
                [cost, acc], feed_dict={x: batch[0], y_: batch[1]})
            test_loss, test_acc = sess.run(
                [cost, acc], feed_dict={
                    x: mnist.test.images, y_: mnist.test.labels})
            print('iteration %d train loss %f train acc %f \
                   test loss %f test acc %f' % (i, train_loss, train_acc,
                                                test_loss, test_acc))
    print('use time %f' % (time.time() - beg))
