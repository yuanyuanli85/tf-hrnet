import tensorflow as tf
import re

FLAGS = tf.app.flags.FLAGS


def setup_lrn_rate_piecewise_constant(global_step, lrn_rate_init, batch_size, idxs_epoch, decay_rates):
    """Setup the learning rate with piecewise constant strategy.

    Args:
    * global_step: training iteration counter
    * batch_size: number of samples in each mini-batch
    * idxs_epoch: indices of epoches to decay the learning rate
    * decay_rates: list of decaying rates

    Returns:
    * lrn_rate: learning rate
    """

    nb_epochs_rat = 1.0
    batch_size_norm = 256

    # adjust interval endpoints w.r.t. FLAGS.nb_epochs_rat
    idxs_epoch = [idx_epoch * nb_epochs_rat for idx_epoch in idxs_epoch]

    # setup learning rate with the piecewise constant strategy
    lrn_rate_init = lrn_rate_init * batch_size / batch_size_norm
    nb_batches_per_epoch = float(FLAGS.nb_smpls_train) / batch_size
    bnds = [int(nb_batches_per_epoch * idx_epoch) for idx_epoch in idxs_epoch]
    vals = [lrn_rate_init * decay_rate for decay_rate in decay_rates]
    lrn_rate = tf.train.piecewise_constant(global_step, bnds, vals)

    return lrn_rate


def get_global_step_from_ckpt(ckptpath):
    step = re.findall("model.ckpt-(\d+)", ckptpath)
    return int(step[0])
