import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import layers
from net.head import ClsHead
from net.stage import HRStage
from net.front import HRFront
from collections import Counter
from utils.config import load_net_cfg_from_file
import functools
from tensorflow.python.ops.init_ops import VarianceScaling


def he_normal_fanout(seed=None):
  """He normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with `stddev = sqrt(2 / fan_out)`
  where `fan_in` is the number of input units in the weight tensor.
  To keep aligned with official implementation
  """
  return VarianceScaling(
      scale=2., mode="fan_out", distribution="truncated_normal", seed=seed)


class HRNet():

    def __init__(self, cfgfile):
        self.stages = []
        self._load_cfg(cfgfile)
        self._build_components()

    def forward_train(self, train_input):

        batch_norm_params = {'epsilon': 1e-5,
                             'scale': True,
                             'is_training': True,
                             'updates_collections': ops.GraphKeys.UPDATE_OPS}

        with slim.arg_scope([layers.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=he_normal_fanout(),
                                weights_regularizer=slim.l2_regularizer(self.cfg['NET']['weight_l2_scale'])):
                final_logit = self._forward(train_input)

        return final_logit

    def forward_eval(self, eval_input):

        batch_norm_params = {'epsilon': 1e-5,
                             'scale': True,
                             'is_training': False,
                             'updates_collections': ops.GraphKeys.UPDATE_OPS}

        with slim.arg_scope([layers.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d],
                                weights_regularizer=slim.l2_regularizer(self.cfg['NET']['weight_l2_scale'])):
                final_logit = self._forward(eval_input)

        return final_logit

    def model_summary(self):

        cnt = Counter()
        ops = ['ResizeNearestNeighbor', 'Relu', 'Conv2D']

        for op in tf.get_default_graph().get_operations():
            if op.type in ops:
                cnt[op.type] += 1

        print(cnt)

    def calc_loss(self, labels, outputs, trainable_vars):

        loss = tf.losses.softmax_cross_entropy(labels, outputs)
        l2_loss = tf.add_n([tf.nn.l2_loss(var) * self.cfg['NET']['weight_l2_scale'] for var in trainable_vars])
        loss += l2_loss

        targets = tf.argmax(labels, axis=1)
        acc_top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(outputs, targets, 1), tf.float32))
        acc_top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(outputs, targets, 5), tf.float32))

        metrics = {'acc_top1': acc_top1, 'acc_top5': acc_top5, 'l2_loss': l2_loss}

        return loss, metrics

    def _build_components(self):

        front = HRFront(num_channels=self.cfg['FRONT']['num_channels'],
                        bottlenect_channels=self.cfg['FRONT']['bottlenect_channels'],
                        output_channels=[i * self.cfg['FRONT']['output_channels'] for i in range(1, 3)],
                        num_blocks=self.cfg['FRONT']['num_blocks'])
        self.stages.append(front)

        num_stages = self.cfg['NET']['num_stages']
        for i in range(num_stages):
            _key = 'S{}'.format(i + 2)
            _stage = HRStage(stage_id=i + 2,
                             num_modules=self.cfg[_key]['num_modules'],
                             num_channels=self.cfg['NET']['num_channels'],
                             num_blocks=self.cfg[_key]['num_blocks'],
                             num_branches=self.cfg[_key]['num_branches'],
                             last_stage=True if i == num_stages - 1 else False)

            self.stages.append(_stage)

        clshead = ClsHead(base_channel=self.cfg['HEAD']['base_channel'],
                          num_branches=self.cfg['HEAD']['num_branches'],
                          cls_num=self.cfg['HEAD']['cls_num'],
                          fc_channel=self.cfg['HEAD']['fc_channel'])

        self.stages.append(clshead)

    def _forward(self, input):

        _out = input
        for stage in self.stages:
            _out = stage.forward(_out)

        return _out

    def _load_cfg(self, cfgfile):
        self.cfg = load_net_cfg_from_file(cfgfile)


    def _get_num_parameters(self):
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        var_size = 0
        for _var in vars:
            _size = functools.reduce(lambda a, b : a*b , _var.shape)
            var_size += _size
        return var_size
