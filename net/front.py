import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import batch_norm
from net.layers import *


class HRFront():

    def __init__(self, num_channels, bottlenect_channels, output_channels, num_blocks):
        self.scope = 'HR_FRONT'
        self.num_channels = num_channels
        self.bottleneck_channels = bottlenect_channels
        self.output_channels = output_channels
        self.num_blocks = num_blocks

    def forward(self, input):
        with tf.variable_scope(self.scope):
            # conv1 + bn1 + relu1
            _out = slim.conv2d(input, num_outputs=self.num_channels, kernel_size=[3, 3],
                               stride=2, activation_fn=tf.nn.relu, normalizer_fn=batch_norm)

            # conv2 + bn2 + relu2
            _out = slim.conv2d(_out, num_outputs=self.num_channels, kernel_size=[3, 3],
                               stride=2, activation_fn=tf.nn.relu, normalizer_fn=batch_norm)

            # bottlenect
            for i in range(self.num_blocks):
                _out = bottleneck_block(_out, planes=self.bottleneck_channels,
                                        scope='_BN' + str(i), downsamplefn=trans_block if i == 0 else None)

            # one 3x3 keep same resolution and one 3x3 to 1/2x resolution
            _same_res = slim.conv2d(_out, num_outputs=self.output_channels[0], kernel_size=[3, 3],
                                    stride=1, activation_fn=tf.nn.relu, normalizer_fn=batch_norm)

            _half_res = slim.conv2d(_out, num_outputs=self.output_channels[1], kernel_size=[3, 3],
                                    stride=2, activation_fn=tf.nn.relu, normalizer_fn=batch_norm)

        return [_same_res, _half_res]
