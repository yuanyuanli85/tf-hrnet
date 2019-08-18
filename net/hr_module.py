import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import batch_norm
from net.layers import *
from net.utils import *
from collections import defaultdict


class HRModule():

    def __init__(self, module_id, num_branches, num_channels, num_blocks, multi_scale_output, scope):
        '''

        :param module_id:
        :param num_branches:
        :param num_channels:
        :param multi_scale_output:
        '''
        self.scope = scope + '_M{}'.format(module_id)
        self.num_branches = num_branches
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.num_outputs = self.num_branches + 1 if multi_scale_output else self.num_branches
        self.multi_scale_output = multi_scale_output
        self.downfn = downsample_block
        self.upfn = upsample_block

    def forward(self, inputs):
        assert len(inputs) == self.num_branches, \
            "input_channel {} must to be same as num_branches {}".format(len(inputs), self.num_branches)

        inter_features = []
        for i, _input in enumerate(inputs):
            _output_per_resolution = self.build_network_per_resolution(_input, channels=self.num_channels * pow(2, i), \
                                                                       scope=self.scope + '_B' + str(i))
            inter_features.append(_output_per_resolution)

        outputs = self.fuse_multi_resolution(inter_features)

        return outputs

    def build_network_per_resolution(self, input, channels, scope):

        _out = input
        for i in range(self.num_blocks):
            _out = basic_residual_block(_out, channels, scope=scope + '_C' + str(i))

        return _out

    def __get_dwsample_features(self, features):

        def insert_dwfn(input, channels, src, dst, dwfn):
            out = input
            for i in range(dst - src):
                scope_name = self.scope + '_D' + str(src) + '_X' + str(i) + '_D' + str(dst)
                if i == dst - src - 1:
                    # for last downsample, no relu
                    out = dwfn(out, channels, scope_name, has_relu=False)
                else:
                    # for inter downsample layer, keep using input channel
                    # this can save number of parameters
                    out = dwfn(out, input.shape[-1], scope_name, has_relu=True)
            return out

        output_channels = [self.num_channels * pow(2, i) for i in range(self.num_outputs)]

        # create downsample fn matrix
        dwmatrix = create_downsample_fn_matrix(self.num_branches, features, self.num_branches, output_channels)

        dwfeatures = defaultdict(list)
        for key, value in dwmatrix.items():
            src, dst = key
            _input = value['input']
            _channels = value['outchannel']
            _out = insert_dwfn(_input, _channels, src, dst, self.downfn)
            dwfeatures[dst].append(_out)

        return dwfeatures

    def __get_upsample_features(self, features):

        output_channels = [self.num_channels * pow(2, i) for i in range(self.num_outputs)]

        # create downsample fn matrix
        upmatrix = create_upsample_fn_matrix(self.num_branches, features, self.num_outputs, output_channels)

        upfeatures = defaultdict(list)

        for key, value in upmatrix.items():
            src, dst = key
            _input = value['input']
            _channels = value['outchannel']
            _out = self.upfn(_input, 2 ** (src - dst), _channels, self.scope + '_U' + str(src) + 'to' + str(dst))
            upfeatures[dst].append(_out)

        return upfeatures

    def fuse_multi_resolution(self, features):
        assert len(features) == self.num_branches, \
            "outputs {} feed to fuse_multi_resolution must to be same as num_branches {}".format(len(features),
                                                                                                 self.num_branches)

        dwfeatrues = self.__get_dwsample_features(features)
        upfeatures = self.__get_upsample_features(features)

        origfeatures = defaultdict(list)
        for i, _orgfeature in enumerate(features):
            origfeatures[i].append(_orgfeature)

        output_features = add_layers(origfeatures, dwfeatrues, upfeatures, self.num_branches)

        if self.multi_scale_output:
            # an extra downsampling
            input = output_features[-1]
            output = self.downfn(input, planes=self.num_channels * (2 ** (self.num_outputs - 1)),
                                 scope=self.scope + '_D_TAIL' + str(self.num_outputs), has_relu=True)

            output_features.append(output)

        return output_features
