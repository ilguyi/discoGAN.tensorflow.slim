from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time

import tensorflow as tf

import discoGAN_model as disco
import data as data

slim = tf.contrib.slim


##################
# Training Flags #
##################
tf.app.flags.DEFINE_string('train_dir',
                           '',
                           'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer('max_steps',
                            10000,
                            'The maximum number of training steps.')
tf.app.flags.DEFINE_integer('save_steps',
                            5000,
                            'The step per saving model.')

#################
# Dataset Flags #
#################
tf.app.flags.DEFINE_integer('batch_size',
                            32,
                            'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('image_size',
                            64,
                            'The size of images.')

########################
# CelebA Dataset Flags #
########################
tf.app.flags.DEFINE_string('style_A',
                           None,
                           'Style for celebA dataset.'
                           'Could be any attributes in celebA (Young, Male, Blond_Hair, Wearing_Hat ...)')
tf.app.flags.DEFINE_string('style_B',
                           None,
                           'Style for celebA dataset.'
                           'Could be any attributes in celebA (Young, Male, Blond_Hair, Wearing_Hat ...)')
tf.app.flags.DEFINE_string('constraint',
                           None,
                           'Constraint for celebA dataset.'
                           'Only images satisfying this constraint is used.'
                           'For example, if --constraint=Male, and --constraint_type=1,'
                           'only male images are used for both style/domain.')
tf.app.flags.DEFINE_string('constraint_type',
                           None,
                           'Used along with --constraint.'
                           'If --constraint_type=1, only images satisfying the constraint are used.'
                           'If --constraint_type=-1, only images not satisfying the constraint are used.')
tf.app.flags.DEFINE_boolean('is_test',
                            False,
                            'Whether train sets or test set.')
tf.app.flags.DEFINE_integer('n_test',
                            200,
                            'Number of test data.')


########################
# Learning rate policy #
########################
tf.app.flags.DEFINE_float('initial_learning_rate',
                          0.0002,
                          'Initial learning rate.')
#tf.app.flags.DEFINE_float('num_epochs_per_decay',
#                          100000,
#                          'Epochs after which learning rate decays.')
#tf.app.flags.DEFINE_float('learning_rate_decay_factor',
#                          0.5,
#                          'Learning rate decay factor.')

######################
# Optimization Flags #
######################
tf.app.flags.DEFINE_float('adam_beta1',
                          0.9,
                          'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float('adam_beta2',
                          0.999,
                          'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('adam_epsilon',
                          1e-08,
                          'Epsilon term for the optimizer.')

########################
# Moving average decay #
########################
tf.app.flags.DEFINE_float('MOVING_AVERAGE_DECAY',
                          0.9999,
                          'Moving average decay.')

FLAGS = tf.app.flags.FLAGS


def main(_):

  if tf.gfile.Exists(FLAGS.train_dir):
    raise ValueError('This folder already exists.')
  tf.gfile.MakeDirs(FLAGS.train_dir)

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
                          FLAGS.train_dir,
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
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

    print('complete training...')



if __name__ == '__main__':
  tf.app.run()
