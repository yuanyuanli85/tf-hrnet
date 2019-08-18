from collections import defaultdict
import functools
import tensorflow as tf


def create_downsample_fn_matrix(num_branches, inputs, num_output_branches, output_channels):
    '''

    :param num_branches: number of input branches
    :param num_output_branches:  number of output branches
    :return: matrix[num_branches,num_output_branches], None if not needed.
    '''

    fn_matrix = {}
    for i in range(num_branches):
        for j in range(num_output_branches):
            if j > i:
                fn_matrix[i, j] = {'input': inputs[i], 'outchannel': output_channels[j]}

    return fn_matrix


def create_upsample_fn_matrix(num_branches, inputs, num_output_branches, output_channels):
    '''

    :param num_branches: number of input branches
    :param num_output_branches:  number of output branches
    :return: matrix[num_branches][num_output_branches], None if not needed.
    '''

    fn_matrix = {}
    for i in range(num_branches):
        for j in range(num_output_branches):
            if j < i:
                fn_matrix[i, j] = {'input': inputs[i], 'outchannel': output_channels[j]}

    return fn_matrix


def add_layers(origfeatures, dwfeatrues, upfeatures, nums_output):
    '''

    :param origfeatures:
    :param dwfeatrues:
    :param upfeatures:
    :param nums_output:
    :return:
    '''

    _temp = defaultdict(list)
    for i in range(nums_output):
        for featuremaps in [origfeatures, dwfeatrues, upfeatures]:
            if i in featuremaps.keys():
                _temp[i].extend(featuremaps[i])

    outlist = []
    for i in range(nums_output):
        fmlist = _temp[i]
        add = functools.reduce(lambda a, b: a + b, fmlist)
        # for each elemwise add, go through relu
        xrelu = tf.nn.relu(add)
        outlist.append(xrelu)
    return outlist
