"""Configuration
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


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
tf.app.flags.DEFINE_float('l2_decay',
                          0.0001,
                          'L2 regularization factor for the optimizer.')

########################
# Moving average decay #
########################
tf.app.flags.DEFINE_float('MOVING_AVERAGE_DECAY',
                          0.9999,
                          'Moving average decay.')

####################
# Checkpoint Flags #
####################
tf.app.flags.DEFINE_string('checkpoint_dir',
                           '',
                           'Directory where to read model checkpoints.')
tf.app.flags.DEFINE_integer('checkpoint_step',
                            -1,
                            'The step you want to read model checkpoints.'
                            '-1 means the latest model.')
tf.app.flags.DEFINE_boolean('is_all_checkpoints',
                            False,
                            'Whether translate image in all checkpoints or one checkpoint.')


FLAGS = tf.app.flags.FLAGS



def hyperparameters_dir(input_dir):
  hp_dir = os.path.join(input_dir, FLAGS.style_A)
  if FLAGS.style_B:
    hp_dir = os.path.join(hp_dir, FLAGS.style_B)
  if FLAGS.constraint:
    hp_dir = os.path.join(hp_dir, FLAGS.constraint)
  if FLAGS.constraint_type:
    hp_dir = os.path.join(hp_dir, FLAGS.constraint_type)
  hp_dir = os.path.join(hp_dir, '%.1E' % FLAGS.initial_learning_rate)
  print(hp_dir)

  return hp_dir

