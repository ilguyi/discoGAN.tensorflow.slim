from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time

import tensorflow as tf

import configuration
import discoGAN_model as disco
import data as data


FLAGS = tf.app.flags.FLAGS


def main(_):

  with tf.Graph().as_default():

    # Build the model.
    model = disco.DiscoGAN(mode="train")
    model.build()

    # No decay learning rate
    learning_rate = tf.constant(FLAGS.initial_learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    # Create an optimizer that performs gradient descent for Discriminator.
    opt_D = tf.train.AdamOptimizer(
                learning_rate,
                beta1=FLAGS.adam_beta1,
                beta2=FLAGS.adam_beta2,
                epsilon=FLAGS.adam_epsilon)

    # Create an optimizer that performs gradient descent for Discriminator.
    opt_G = tf.train.AdamOptimizer(
                learning_rate,
                beta1=FLAGS.adam_beta1,
                beta2=FLAGS.adam_beta2,
                epsilon=FLAGS.adam_epsilon)

    # Minimize optimizer
    opt_D_op = opt_D.minimize(model.loss_Discriminator,
                              global_step=model.global_step,
                              var_list=model.D_vars)
    opt_G_op = opt_G.minimize(model.loss_Generator,
                              var_list=model.G_vars)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.MOVING_AVERAGE_DECAY, model.global_step)
    variables_to_average = tf.trainable_variables()
    variables_averages_op = variable_averages.apply(variables_to_average)

    # Batch normalization update
    batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    batchnorm_updates_op = tf.group(*batchnorm_updates)

    train_op = tf.group(opt_D_op, opt_G_op, variables_averages_op,
                        batchnorm_updates_op)


    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

    # Build the summary operation
    summary_op = tf.summary.merge_all()

    # train_dir path in each the combination of hyper-parameters
    train_dir = configuration.hyperparameters_dir(FLAGS.train_dir)

    # Training with tf.train.Supervisor.
    sv = tf.train.Supervisor(logdir=train_dir,
                             summary_op=None,     # Do not run the summary services
                             saver=saver,
                             save_model_secs=0,   # Do not run the save_model services
                             init_fn=None)        # Not use pre-trained model
    # Start running operations on the Graph.
    with sv.managed_session() as sess:
      tf.logging.info('Start Session')

      # Start the queue runners.
      sv.start_queue_runners(sess=sess)
      tf.logging.info('Starting Queues.')

      # Read dataset
      data_A, data_B = data.get_data()
      data_size = min( len(data_A), len(data_B) )

      pre_epochs = 0.0

      for step in range(FLAGS.max_steps):
        start_time = time.time()
        if sv.should_stop():
          break

        epochs = step * FLAGS.batch_size / data_size
        A_path, B_path = data.get_batch(FLAGS.batch_size, data_A, data_B, pre_epochs, epochs, step, data_size)

        images_A = data.read_images(A_path, None, FLAGS.image_size)
        images_B = data.read_images(B_path, None, FLAGS.image_size)

        feed_dict = {model.images_A: images_A,
                     model.images_B: images_B}
        _, _global_step, loss_D, loss_G = sess.run([train_op,
                                                    sv.global_step,
                                                    model.loss_Discriminator,
                                                    model.loss_Generator],
                                                    feed_dict=feed_dict)

        pre_epochs = epochs
        duration = time.time() - start_time

        # Monitoring training situation in console.
        if _global_step % 10 == 0:
          examples_per_sec = FLAGS.batch_size / float(duration)
          print("Epochs: %.2f global_step: %d  loss_D: %f loss_G: %f (%.1f examples/sec; %.3f sec/batch)"
                    % (epochs, _global_step, loss_D, loss_G, examples_per_sec, duration))

        # Save the model summaries periodically.
        if _global_step % 200 == 0:
          summary_str = sess.run(summary_op, feed_dict=feed_dict)
          sv.summary_computed(sess, summary_str)

        # Save the model checkpoint periodically.
        if _global_step % FLAGS.save_steps == 0:
          tf.logging.info('Saving model with global step %d to disk.' % _global_step)
          sv.saver.save(sess, sv.save_path, global_step=sv.global_step)

    tf.logging.info('complete training...')



if __name__ == '__main__':
  tf.app.run()
