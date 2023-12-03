import tensorflow as tf

def discriminator_nn(inputs, embedding_dim, task, initializer):
    height = inputs.shape[0]
    input_layer = tf.reshape(inputs, [-1, height, embedding_dim, 1])
    conv1_w = tf.get_variable('conv1_w', [3, 5, 1, 1], initializer=initializer)
    conv1 = tf.nn.conv2d(input_layer, conv1_w, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(conv1)
    pool2_flat = tf.reshape(relu1, [-1, height * embedding_dim])
    weight1 = tf.get_variable('weight1', [height * embedding_dim, 1024], initializer=initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01))
    bias1 = tf.get_variable('bias3', [1024], initializer=tf.constant_initializer(0.0))
    hidden1 = tf.nn.relu(tf.matmul(pool2_flat, weight1) + bias1)
    weight2 = tf.get_variable('weight2', [1024, embedding_dim], initializer=initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01))
    bias2 = tf.get_variable('bias2', [embedding_dim], initializer=tf.constant_initializer(0.0))
    output = tf.nn.sigmoid(tf.matmul(hidden1, weight2) + bias2)
    return output