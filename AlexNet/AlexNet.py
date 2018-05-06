import tensorflow as tf
import  numpy as np

KEEPPRO = 0.5
CLASSNUM = 2

#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)#平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)#标准差
        tf.summary.scalar('max', tf.reduce_max(var))#最大值
        tf.summary.scalar('min', tf.reduce_min(var))#最小值
        tf.summary.histogram('histogram', var)#直方图


#获取变量
def _get_variable(name,shape,initializer, regularizer=None,dtype='float',trainable=True):
    collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    var = tf.get_variable(name,
                          shape=shape,
                          initializer=initializer,
                          dtype=dtype,
                          regularizer=regularizer,
                          collections=collections,
                          trainable=trainable)

    return var

#最大池化层
def maxPoolLayer(x, size, stride, name, padding="SAME"):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1],
                          strides=[1, stride, stride, 1], padding=padding, name=name)
#平均池化层
def avgPoolLayer(x, size, stride,name, padding='SAME'):
    return tf.nn.avg_pool(x,ksize=[1, size, size, 1],
                          strides=[1, stride, stride, 1], padding=padding, name=name)
#dropout层，应用在全连接层，防止过拟合
def dropout(x, dropPro, name=None):
    return tf.nn.dropout(x, dropPro, name)
#局部响应归一化，防止过拟合
def LRN(x, dr, alpha, beta, name=None, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=dr, alpha=alpha,
                                              beta=beta, bias=bias, name=name)
#全连接层
def fcLayer(x, input, output, isRelu, name):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[input, output], dtype="float")
        b = tf.get_variable("b", [output], dtype="float")
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
        if isRelu:
            return tf.nn.relu(out)
        else:
            return out
#卷积层
def Convolution(x, size, stride, filters_out,name, wd=0.0,paddingFlag = "VALID", weight_initializer=None, bias_initializer=None):
    with tf.variable_scope(name) as scope:
        filters_in = x.get_shape()[-1]
        stddev = 1. / tf.sqrt(tf.cast(filters_out, tf.float32))
        if weight_initializer is None:
            weight_initializer = tf.random_uniform_initializer(minval=-stddev, maxval=stddev, dtype=tf.float32)
        if bias_initializer is None:
            bias_initializer = tf.random_uniform_initializer(minval=-stddev, maxval=stddev, dtype=tf.float32)
        shape = [size, size, filters_in, filters_out]
        # weights = _get_variable('weights',shape, weight_initializer, tf.contrib.layers.l2_regularizer(wd))
        weights = _get_variable('weights', shape, weight_initializer)
        variable_summaries(weights)
        conv = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=paddingFlag)
        biases = _get_variable('biases', [filters_out], bias_initializer)
        variable_summaries(biases)
        return tf.nn.bias_add(conv, biases)
#将维度展成1维
def flatten(x):
    shape = x.get_shape().as_list()
    dim = 1
    for i in range(1,len(shape)):
        dim*=shape[i]
    return tf.reshape(x, [-1, dim]),dim
#网络推理
def alexnet(input):
    with tf.name_scope("layer_1"):
        conv1 = Convolution(input, 11, 4, 96, "conv1", paddingFlag ="SAME")
        lrn1 = LRN(conv1, 2, 2e-05, 0.75, "lrn1")
        pool1 = maxPoolLayer(lrn1, 3, 2, "pool1", "SAME")
    with tf.name_scope("layer_2"):
        conv2 = Convolution(pool1, 5, 1, 256, "conv2", paddingFlag ="SAME")
        lrn2 = LRN(conv2, 2, 2e-05, 0.75, "lrn2")
        pool2 = maxPoolLayer(lrn2, 3, 2, "pool2", "SAME")

    with tf.name_scope("layer_3"):
        conv3 = Convolution(pool2, 3, 1, 384, "conv3",paddingFlag ="SAME")

    with tf.name_scope("layer_4"):
        conv4 = Convolution(conv3, 3, 1, 384, "conv4",paddingFlag ="SAME")

    with tf.name_scope("layer_5"):
        conv5 = Convolution(conv4, 3, 1, 256, "conv5", paddingFlag ="SAME")
        pool5 = maxPoolLayer(conv5, 3, 2, "pool5","SAME")
        with tf.name_scope("flatten"):
            fcIn,dim = flatten(pool5)
    with tf.name_scope('layer_6'):
        fc1 = fcLayer(fcIn, dim, 1024, True, "fc6")
        dropout1 = dropout(fc1, KEEPPRO)
    with tf.name_scope('layer_7'):
        fc2 = fcLayer(dropout1, 1024, 1024, True, "fc7")
        dropout2 = dropout(fc2, KEEPPRO)
    with tf.name_scope('layer_8'):
        fc3 = fcLayer(dropout2, 1024, CLASSNUM, True, "fc8")

    return fc3