import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import batch_norm


def basic_residual_block(input, planes, scope, kernel_size=3, stride=1, downsamplefn=None):
    '''

    :param planes:   kernel number , output channel
    :param stride:   stride
    :param downsample:  downsample fn
    :return:
    '''

    residual = input

    with tf.variable_scope(scope):
        _out = slim.conv2d(input, num_outputs=planes, kernel_size=[kernel_size, kernel_size],
                           stride=stride, activation_fn=tf.nn.relu, normalizer_fn=batch_norm)

        _out = slim.conv2d(_out, num_outputs=planes, kernel_size=[kernel_size, kernel_size],
                           stride=stride, activation_fn=None, normalizer_fn=batch_norm)

        if downsamplefn is not None:
            residual = downsamplefn(residual)

        out = _out + residual
        out = tf.nn.relu(out)

    return out


def bottleneck_block(input, planes, scope, stride=1, downsamplefn=None):
    '''

    :param planes:   kernel number , output channel
    :param stride:   stride
    :param downsample:  downsample fn
    :return:
    '''

    expansion = 4
    residual = input

    with tf.variable_scope(scope):
        # conv1x1 + bn1 + relu
        _out = slim.conv2d(input, num_outputs=planes // expansion, kernel_size=[1, 1],
                           stride=1, activation_fn=tf.nn.relu, normalizer_fn=batch_norm)

        # conv2 3x3 + bn2 + relu
        _out = slim.conv2d(_out, num_outputs=planes // expansion, kernel_size=[3, 3],
                           stride=stride, activation_fn=tf.nn.relu, normalizer_fn=batch_norm)

        # conv3 1x1 + bn3
        _out = slim.conv2d(_out, num_outputs=planes, kernel_size=[1, 1],
                           stride=1, activation_fn=None, normalizer_fn=batch_norm)

        if downsamplefn is not None:
            residual = downsamplefn(residual, planes)

        out = _out + residual
        out = tf.nn.relu(out)

    return out


def trans_block(input, planes):
    # conv 1x1 +  bn
    _out = slim.conv2d(input, num_outputs=planes, kernel_size=[1, 1],
                       stride=1, activation_fn=None, normalizer_fn=batch_norm)

    return _out


def upsample_block(input, ratio, planes, scope):
    '''

    :param input:
    :param planes:
    :param scope:
    :param kernel_size:
    :param stride:
    :return:
    '''

    with tf.variable_scope(scope):
        _out = slim.conv2d(input, num_outputs=planes, kernel_size=[1, 1], stride=1,
                           activation_fn=None, normalizer_fn=batch_norm, padding='SAME')
        shape = _out.shape
        _out = tf.image.resize_nearest_neighbor(_out, (shape[1] * ratio, shape[1] * ratio))

    return _out


def downsample_block(input, planes, scope, has_relu):
    '''

    :param input:
    :param planes:
    :param scope:
    :return:
    '''

    with tf.variable_scope(scope):
        _out = slim.conv2d(input, num_outputs=planes, kernel_size=[3, 3], stride=2,
                           activation_fn=tf.nn.relu if has_relu else None,
                           normalizer_fn=batch_norm, padding='SAME')

    return _out


def conv1x1_block(input, planes, scope, has_relu):
    '''

    :param input:
    :param planes:
    :param scope:
    :return:
    '''

    with tf.variable_scope(scope):
        _out = slim.conv2d(input, num_outputs=planes, kernel_size=[1, 1], stride=1,
                           activation_fn=tf.nn.relu if has_relu else None,
                           normalizer_fn=batch_norm, padding='SAME')

    return _out
