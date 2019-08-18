import tensorflow as tf
from trainer.trainer import Trainer
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool('enbl_multi_gpu', False, 'Enable training with multiple gpus')
tf.app.flags.DEFINE_string('data_path', './data/tfrecord', 'path to data tfrecords')
tf.app.flags.DEFINE_string('net_cfg', './cfgs/w30_s4.cfg', 'config file of network')
tf.app.flags.DEFINE_bool('eval_only', False, 'Eval mode')
tf.app.flags.DEFINE_bool('resume_training', False, 'resume training')


def main(unused_argv):
    """Main entry.

    Args:
    * unused_argv: unused arguments (after FLAGS is parsed)
    """

    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.enbl_multi_gpu:
        mgw.init()

    trainer = Trainer(data_path=FLAGS.data_path, netcfg=FLAGS.net_cfg)

    trainer.build_graph(is_train=True)
    trainer.build_graph(is_train=False)

    if FLAGS.eval_only:
        trainer.eval()
    else:
        trainer.train()


if __name__ == '__main__':
    tf.app.run()
