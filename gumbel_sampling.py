from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def get_gumbel(shape, eps=1e-8):
    """Generate samples from a gumbel distribution."""

    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U+eps) + eps)


def main():

    use_sampling = True
    K = 5
    target = np.asarray([0.05, 0.1, 0.15, 0.2, 0.5], dtype=np.float32)
    global_step = tf.train.get_or_create_global_step()
    temperature = tf.train.exponential_decay(learning_rate=1.1,
                                             global_step=global_step,
                                             decay_steps=100,
                                             decay_rate=0.99,
                                             staircase=True)
    param = tf.get_variable(
                name="param",
                shape=(1, K),
                initializer=tf.constant_initializer(.0))
    if use_sampling:
        logits = param - tf.log(tf.reduce_sum(tf.exp(param), axis=-1, keep_dims=True) + (1e-8))
        gumbel_noise = get_gumbel(logits.shape)
        y_soft = tf.nn.softmax((logits + gumbel_noise) / temperature)
        y_hard = tf.cast(tf.equal(y_soft, tf.reduce_max(y_soft, -1, keep_dims=True)),
                         y_soft.dtype)
        y = tf.stop_gradient(y_hard - y_soft) + y_soft
    else:
        y_soft = tf.nn.softmax(param)
        y = y_soft
    # kl
    losses = tf.reduce_sum(y * (tf.log(tf.clip_by_value(y, 1e-8, 1-(1e-8)))-tf.log(target)), -1)
    loss = tf.reduce_mean(losses, 0)
    opt = tf.train.AdamOptimizer(0.01, name="model_opt")
    update_op = opt.minimize(loss, global_step=global_step, var_list=[param])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) 
    sess.run(tf.local_variables_initializer())
    print(sess.run(param))
    print(sess.run(tf.nn.softmax(param)))
    while True:
        gs, temp, y_val, loss_value, _ = sess.run(
            [global_step, temperature, y, loss, update_op])
        print("{}: temp={} y={} loss={}".format(gs, temp, y_val, loss_value))
        if temp < 0.1:
            break
    print(sess.run(param))
    if use_sampling:
        print(sess.run(tf.nn.softmax(param/0.1)))
    else:
        print(sess.run(tf.nn.softmax(param)))


if __name__=="__main__":
    main()
