from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import tensorflow as tf

import ops

slim = tf.contrib.slim
layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope

FLAGS = tf.app.flags.FLAGS


class Generator(object):
  """Generator setup.

  Args:
    images: A float32 scalar Tensor of real images from one domain
    scope: name scope
    extra_layers: boolean whether use conv5 layer (1 x 1 x 100 dim) or not
    reuse: reuse flag

  Returns: 
    A float32 scalar Tensor of generated images from one domain to another domain
  """
  def __init__(self, scope=None, extra_layers=False):
    self.scope = scope
    self.extra_layers = extra_layers

  def __call__(self, images, is_training=True, reuse=False):
    with tf.variable_scope(self.scope + '/Generator') as scope:
      if reuse:
        scope.reuse_variables()

      batch_norm_params = {'decay': 0.999,
                           'epsilon': 0.001,
                           'is_training': is_training,
                           'scope': 'batch_norm'}
      with arg_scope([layers.conv2d, layers.conv2d_transpose],
                      kernel_size=[4, 4],
                      stride=[2, 2],
                      normalizer_fn=layers.batch_norm,
                      normalizer_params=batch_norm_params,
                      weights_regularizer=layers.l2_regularizer(0.0001, scope='l2_decay')):

        # Encoder part (sequence of conv2d)
        with arg_scope([layers.conv2d], activation_fn=ops.leakyrelu):
          # inputs: 64 x 64 x 3
          self.conv1 = layers.conv2d(inputs=images,
                                     num_outputs=64 * 1,
                                     normalizer_fn=None,
                                     biases_initializer=None,
                                     scope='conv1')

          # conv1: 32 x 32 x (64 * 1)
          self.conv2 = layers.conv2d(inputs=self.conv1,
                                     num_outputs=64 * 2,
                                     scope='conv2')

          # conv2: 16 x 16 x (64 * 2)
          self.conv3 = layers.conv2d(inputs=self.conv2,
                                     num_outputs=64 * 4,
                                     scope='conv3')

          # conv3: 8 x 8 x (64 * 4)
          self.conv4 = layers.conv2d(inputs=self.conv3,
                                     num_outputs=64 * 8,
                                     scope='conv4')

          if self.extra_layers == True:
            # conv4: 4 x 4 x (64 * 8)
            self.conv5 = layers.conv2d(inputs=self.conv4,
                                       num_outputs=100,
                                       stride=[1, 1],
                                       padding='VALID',
                                       scope='conv5')

            # Decoder part
            # conv5: 1 x 1 x (100)
            self.convt1 = layers.conv2d_transpose(inputs=self.conv5,
                                                  num_outputs=64 * 8,
                                                  padding='VALID',
                                                  scope='convt1')
          else:
            self.convt1 = self.conv4


        # Decoder part (sequence of conv2d_transpose)
        # convt1: 4 x 4 x (64 * 8)
        self.convt2 = layers.conv2d_transpose(inputs=self.convt1,
                                              num_outputs=64 * 4,
                                              scope='convt2')

        # convt2: 8 x 8 x (64 * 4)
        self.convt3 = layers.conv2d_transpose(inputs=self.convt2,
                                              num_outputs=64 * 2,
                                              scope='convt3')

        # convt3: 16 x 16 x (64 * 2)
        self.convt4 = layers.conv2d_transpose(inputs=self.convt3,
                                              num_outputs=64 * 1,
                                              scope='convt4')

        # layer4: 32 x 32 x (64 * 1)
        self.convt5 = layers.conv2d_transpose(inputs=self.convt4,
                                              num_outputs=3,
                                              #activation_fn=tf.sigmoid,
                                              activation_fn=tf.tanh,
                                              normalizer_fn=None,
                                              biases_initializer=None,
                                              scope='convt5')

        # output: 64 x 64 x 3
        generated_images = self.convt5

        return generated_images



class Discriminator(object):
  """Discriminator setup.

  Args:
    images: A float32 scalar Tensor of real images from data
    scope: name scope
    reuse: reuse flag

  Returns: 
    logits: A float32 scalar Tensor of dim [batch_size]
  """
  def __init__(self, scope=None):
    self.scope = scope


  def __call__(self, images, reuse=False):
    with tf.variable_scope(self.scope + '/Discriminator') as scope:
      if reuse:
        scope.reuse_variables()

      batch_norm_params = {'decay': 0.999,
                           'epsilon': 0.001,
                           'scope': 'batch_norm'}
      with arg_scope([layers.conv2d],
                      kernel_size=[4, 4],
                      stride=[2, 2],
                      activation_fn=ops.leakyrelu,
                      normalizer_fn=layers.batch_norm,
                      normalizer_params=batch_norm_params,
                      weights_regularizer=layers.l2_regularizer(0.0001, scope='l2_decay')):

        # inputs: 64 x 64 x 3
        self.conv1 = layers.conv2d(inputs=images,
                                   num_outputs=64 * 1,
                                   normalizer_fn=None,
                                   biases_initializer=None,
                                   scope='conv1')

        # conv1: 32 x 32 x (64 * 1)
        self.conv2 = layers.conv2d(inputs=self.conv1,
                                   num_outputs=64 * 2,
                                   scope='conv2')

        # conv2: 16 x 16 x (64 * 2)
        self.conv3 = layers.conv2d(inputs=self.conv2,
                                   num_outputs=64 * 4,
                                   scope='conv3')
        
        # conv3: 8 x 8 x (64 * 4)
        self.conv4 = layers.conv2d(inputs=self.conv3,
                                   num_outputs=64 * 8,
                                   scope='conv4')

        # conv4: 4 x 4 x (64 * 8)
        self.conv5 = layers.conv2d(inputs=self.conv4,
                                    num_outputs=1,
                                    stride=[1, 1],
                                    padding='VALID',
                                    activation_fn=None,
                                    normalizer_fn=None,
                                    biases_initializer=None,
                                    scope='conv5')

        discriminator_logits = self.conv5

        return discriminator_logits



class DiscoGAN(object):
  """Discover Cross-Domain Relations with Generative Adversarial Networks
  implementation based on http://arxiv.org/abs/1703.05192

  "Learning to Discover Cross-Domain Relations
    with Generative Adversarial Networks"
  Taeksoo Kim, Moonsu Cha, Hyunsoo Kim, Jung Kwon Lee, and Jiwon Kim
  """

  def __init__(self, mode):
    """Basic setup.
    """
    assert mode in ["train", "translate"]
    self.mode = mode

    self.batch_size = FLAGS.batch_size

    print('The mode is %s.' % self.mode)
    print('complete initializing model.')


  def read_images_from_placeholder(self):
    # Setup the placeholder of data
    with tf.variable_scope('read_real_images'):
      self.images_A = tf.placeholder(dtype=tf.float32,
                                     shape=[self.batch_size, 64, 64, 3],
                                     name='images_A')
      self.images_B = tf.placeholder(dtype=tf.float32,
                                     shape=[self.batch_size, 64, 64, 3],
                                     name='images_B')


  def build(self):
    # read images from domain A and B
    self.read_images_from_placeholder()

    # Create the Generator class
    generator_A2B = Generator('A2B', extra_layers=False)
    generator_B2A = Generator('B2A', extra_layers=False)
    # Create the Discriminator class
    discriminator_A = Discriminator('A')
    discriminator_B = Discriminator('B')

    # Image generation from one domain to another domain
    self.generated_images_A2B = generator_A2B(self.images_A)  # G_AB
    self.generated_images_B2A = generator_B2A(self.images_B)  # G_BA

    # Image generation from one domain via another domain to original domain
    self.generated_images_A2B2A = generator_B2A(self.generated_images_A2B, reuse=True)  # G_BA
    self.generated_images_B2A2B = generator_A2B(self.generated_images_B2A, reuse=True)  # G_AB


    if self.mode == "train":
      # Discriminate real images by Discriminator()
      self.logits_real_A = discriminator_A(self.images_A)  # D_A
      self.logits_real_B = discriminator_B(self.images_B)  # D_B

      # Discriminate generated (fake) images by Discriminator()
      self.logits_fake_B2A = discriminator_A(self.generated_images_B2A, reuse=True) # D_A
      self.logits_fake_A2B = discriminator_B(self.generated_images_A2B, reuse=True) # D_B

      # Real/Fake GAN Loss (A)
      self.loss_real_A = ops.GANLoss(logits=self.logits_real_A, is_real=True)
      self.loss_fake_A = ops.GANLoss(logits=self.logits_fake_B2A, is_real=False)

      # Real/Fake GAN Loss (B)
      self.loss_real_B = ops.GANLoss(logits=self.logits_real_B, is_real=True)
      self.loss_fake_B = ops.GANLoss(logits=self.logits_fake_A2B, is_real=False)

      # Losses of Discriminator
      self.loss_Discriminator_A = self.loss_real_A + self.loss_fake_A # L_D_A in paper notation
      self.loss_Discriminator_B = self.loss_real_B + self.loss_fake_B # L_D_B in paper notation


      # Losses of GAN Loss (Generator)
      self.loss_GAN_A2B = ops.GANLoss(logits=self.logits_fake_A2B,  # L_GAN_B in paper notation
                                      is_real=True)
      self.loss_GAN_B2A = ops.GANLoss(logits=self.logits_fake_B2A,  # L_GAN_A in paper notation
                                      is_real=True)
      
      # Reconstruction Loss
      self.loss_reconst_A = ops.ReconstructionLoss(self.images_A,
                                                   self.generated_images_A2B2A)
      self.loss_reconst_B = ops.ReconstructionLoss(self.images_B,
                                                   self.generated_images_B2A2B)

      # Losses of Generator
      self.loss_Generator_A2B = self.loss_GAN_A2B + self.loss_reconst_A
      self.loss_Generator_B2A = self.loss_GAN_B2A + self.loss_reconst_B


      # Total loss
      self.loss_Generator = self.loss_Generator_A2B + self.loss_Generator_B2A
      self.loss_Discriminator = self.loss_Discriminator_A + self.loss_Discriminator_B


      # Separate variables for each function
      t_vars = tf.trainable_variables()
        
      self.D_vars = [var for var in t_vars if 'Discriminator' in var.name]
      self.G_vars = [var for var in t_vars if 'Generator' in var.name]

      for var in self.G_vars:
        print(var.name)
      for var in self.D_vars:
        print(var.name)

      # Add summaries.
      # Add loss summaries
      tf.summary.scalar("losses/loss_Discriminator", self.loss_Discriminator)
      tf.summary.scalar("losses/loss_Generator", self.loss_Generator)

      # Add histogram summaries
      for var in self.D_vars:
        tf.summary.histogram(var.op.name, var)
      for var in self.G_vars:
        tf.summary.histogram(var.op.name, var)

      # Add image summaries
      tf.summary.image('domain_A', self.images_A, max_outputs=4)
      tf.summary.image('domain_B', self.images_B, max_outputs=4)
      tf.summary.image('generated_images_A2B', self.generated_images_A2B, max_outputs=4)
      tf.summary.image('generated_images_B2A', self.generated_images_B2A, max_outputs=4)
      tf.summary.image('generated_images_A2B2A', self.generated_images_A2B2A, max_outputs=4)
      tf.summary.image('generated_images_B2A2B', self.generated_images_B2A2B, max_outputs=4)


    print('complete model build.')



  def image_translate(self):
    # read images from domain A and B
    self.read_images_from_placeholder()

    # Create the Generator class
    generator_A2B = Generator('A2B', extra_layers=False)
    generator_B2A = Generator('B2A', extra_layers=False)

    # Image generation from one domain to another domain
    self.generated_images_A2B = generator_A2B(self.images_A, is_training=False)  # G_AB
    self.generated_images_B2A = generator_B2A(self.images_B, is_training=False)  # G_BA

    # Image generation from one domain via another domain to original domain
    self.generated_images_A2B2A = generator_B2A(self.generated_images_A2B, is_training=False, reuse=True)  # G_BA
    self.generated_images_B2A2B = generator_A2B(self.generated_images_B2A, is_training=False, reuse=True)  # G_AB


