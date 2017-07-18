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

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS


def main(_):

  # train_dir path in each the combination of hyper-parameters
  train_dir = configuration.hyperparameters_dir(FLAGS.train_dir)

  if tf.gfile.Exists(train_dir):
    raise ValueError('This folder already exists.')
  tf.gfile.MakeDirs(train_dir)

  with tf.Graph().as_default():

    # Build the model.
    model = disco.DiscoGAN(mode="train")
    model.build()

    # Create global step
    global_step = slim.create_global_step()

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
    opt_op_D = opt_D.minimize(model.loss_Discriminator,
                              global_step=global_step,
                              var_list=model.D_vars)
    opt_op_G = opt_G.minimize(model.loss_Generator,
                              global_step=global_step,
                              var_list=model.G_vars)

    # Track the moving averages of all trainable variables.
    # Note that we maintain a "double-average" of the BatchNormalization
    # global statistics. This is more complicated then need be but we employ
    # this for backward-compatibility with our previous models.
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.MOVING_AVERAGE_DECAY, global_step)

    # Another possibility is to use tf.slim.get_variables().
    variables_to_average = (tf.trainable_variables() +
                            tf.moving_average_variables())
    variables_averages_op = variable_averages.apply(variables_to_average)

    # Batch normalization update
    batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    batchnorm_updates_op = tf.group(*batchnorm_updates)

    train_op = tf.group(opt_op_D, opt_op_G, variables_averages_op,
                        batchnorm_updates_op)

    # Add dependency to compute batchnorm_updates.
    with tf.control_dependencies([variables_averages_op, batchnorm_updates_op]):
      opt_op_D
      opt_op_G

    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)


    # Start running operations on the Graph.
    with tf.Session() as sess:
      # Build an initialization operation to run below.
      init = tf.global_variables_initializer()
      sess.run(init)

      # Start the queue runners.
      tf.train.start_queue_runners(sess=sess)

      # Create a summary writer, add the 'graph' to the event file.
      summary_writer = tf.summary.FileWriter(
                          train_dir,
                          sess.graph)

      # Retain the summaries and build the summary operation
      summary_op = tf.summary.merge_all()

      # Read dataset
      data_A, data_B = data.get_data()
      data_size = min( len(data_A), len(data_B) )

      pre_epochs = 0.0

      for step in range(FLAGS.max_steps+1):
        start_time = time.time()

        epochs = step * FLAGS.batch_size / data_size
        A_path, B_path = data.get_batch(FLAGS.batch_size, data_A, data_B, pre_epochs, epochs, step, data_size)

        images_A = data.read_images(A_path, None, FLAGS.image_size)
        images_B = data.read_images(B_path, None, FLAGS.image_size)

        feed_dict = {model.images_A: images_A,
                     model.images_B: images_B}
        _, loss_D, loss_G = sess.run([train_op,
                                      model.loss_Discriminator,
                                      model.loss_Generator],
                                      feed_dict=feed_dict)

        pre_epochs = epochs
        duration = time.time() - start_time

        #if step % 10 == 0:
        examples_per_sec = FLAGS.batch_size / float(duration)
        print("Epochs: %.2f step: %d  loss_D: %f loss_G: %f (%.1f examples/sec; %.3f sec/batch)"
                  % (epochs, step, loss_D, loss_G, examples_per_sec, duration))
          
        if step % 200 == 0:
          summary_str = sess.run(summary_op, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % FLAGS.save_steps == 0:
          checkpoint_path = os.path.join(train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)


    print('complete training...')



if __name__ == '__main__':
  tf.app.run()
