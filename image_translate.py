from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import cv2

import tensorflow as tf

import discoGAN_model as disco
import data as data

slim = tf.contrib.slim


####################
# Checkpoint Flags #
####################
tf.app.flags.DEFINE_string('checkpoint_path',
                           '',
                           'Directory where to read model checkpoints.')
tf.app.flags.DEFINE_integer('checkpoint_step',
                            -1,
                            'The step you want to read model checkpoints.'
                            '-1 means the latest model.')
tf.app.flags.DEFINE_boolean('is_all_checkpoints',
                            False,
                            'Whether translate image in all checkpoints or one checkpoint.')
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
                            True,
                            'Whether train sets or test set.')
tf.app.flags.DEFINE_integer('n_test',
                            200,
                            'Number of test data.')

########################
# Moving average decay #
########################
tf.app.flags.DEFINE_float('MOVING_AVERAGE_DECAY',
                          0.9999,
                          'Moving average decay.')

FLAGS = tf.app.flags.FLAGS



def make_squared_image(generated_images):
  # Scale from [-1, 1] to [0, 255]
  generated_images += 1.
  generated_images *= 255. * 0.5

  N = len(generated_images)
  black_image = np.zeros(generated_images[0].shape, dtype=np.int32)
  w = int(np.minimum(10, np.sqrt(N)))
  h = int(np.ceil(N / w))

  one_row_image = generated_images[0]
  for j in range(1, w):
    one_row_image = np.concatenate((one_row_image, generated_images[j]), axis=1)
  
  image = one_row_image
  for i in range(1, h):
    one_row_image = generated_images[i*w]
    for j in range(1, w):
      try:
        one_row_image = np.concatenate((one_row_image, generated_images[i*w + j]), axis=1)
      except:
        one_row_image = np.concatenate((one_row_image, black_image), axis=1)
    image = np.concatenate((image, one_row_image), axis=0)

  return image


def merge_images(A, A2B, A2B2A):
  margin = np.ones((A.shape[0], 20, 3)) * 255
  merged_images = np.concatenate((A, margin), axis=1)
  merged_images = np.concatenate((merged_images, A2B), axis=1)
  merged_images = np.concatenate((merged_images, margin), axis=1)
  merged_images = np.concatenate((merged_images, A2B2A), axis=1)

  return merged_images


def ImageWrite(image, name, step):
  r,g,b = cv2.split(image)
  image = cv2.merge([b,g,r])

  filename = 'styleA_%s_styleB_%s_' % (FLAGS.style_A, FLAGS.style_B)
  filename += name
  filename += '_%06.d.jpg' % step
  cv2.imwrite(filename, image)


def run_generator_once(saver, checkpoint_path, model, images_A, images_B):
  print(checkpoint_path)
  start_time = time.time()
  with tf.Session() as sess:
    tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
    saver.restore(sess, checkpoint_path)
    tf.logging.info("Successfully loaded checkpoint: %s",
                    os.path.basename(checkpoint_path))

    generated_images_A2B = model.generated_images_A2B
    generated_images_B2A = model.generated_images_B2A
    generated_images_A2B2A = model.generated_images_A2B2A
    generated_images_B2A2B = model.generated_images_B2A2B
    feed_dict = {model.images_A: images_A,
                 model.images_B: images_B}
    A2B, B2A, \
      A2B2A, B2A2B = sess.run([generated_images_A2B,
                               generated_images_B2A,
                               generated_images_A2B2A,
                               generated_images_B2A2B],
                               feed_dict=feed_dict)

    duration = time.time() - start_time
    print("Loading time: %.3f" % duration)

  return A2B, B2A, A2B2A, B2A2B



def main(_):
  if not FLAGS.checkpoint_path:
    raise ValueError('You must supply the checkpoint_path with --checkpoint_path')

  with tf.Graph().as_default():
    start_time = time.time()

    # Build the DiscoGAN model.
    model = disco.DiscoGAN(mode="translate")
    model.build()

    # Restore the moving average version of the learned variables for image translate.
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    # Set up the Saver for saving and restoring model checkpoints.
    #saver = tf.train.Saver()
    saver = tf.train.Saver(variables_to_restore)

    # Read dataset
    data_A, data_B = data.get_data()
    data_size = min( len(data_A), len(data_B) )
    A_path, B_path = data.get_batch(FLAGS.batch_size, data_A, data_B, 0, 0, 0, data_size)
    images_A = data.read_images(A_path, None, FLAGS.image_size)
    images_B = data.read_images(B_path, None, FLAGS.image_size)


    if FLAGS.is_all_checkpoints:
      # Find all checkpoint_path
      if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_filenames = []
        for filename in os.listdir(FLAGS.checkpoint_path):
          if '.data-00000-of-00001' in filename:
            filename = filename.split(".")[1].split("ckpt-")[1]
            checkpoint_filenames.append(filename)
      else:
        raise ValueError("checkpoint_path must be folder path")

      checkpoint_filenames.sort(key=int)
      for i, filename in enumerate(checkpoint_filenames):
        filename = 'model.ckpt-' + filename
        checkpoint_filenames[i] = filename

      for checkpoint_path in checkpoint_filenames:
        checkpoint_path = os.path.join(FLAGS.checkpoint_path, checkpoint_path)

        A2B, B2A, A2B2A, B2A2B = run_generator_once(saver, checkpoint_path, model, images_A, images_B)

        squared_A = make_squared_image(np.copy(images_A))
        squared_B = make_squared_image(np.copy(images_B))
        squared_A2B = make_squared_image(A2B)
        squared_B2A = make_squared_image(B2A)
        squared_A2B2A = make_squared_image(A2B2A)
        squared_B2A2B = make_squared_image(B2A2B)

        domain_A_images = merge_images(squared_A, squared_A2B, squared_A2B2A)
        domain_B_images = merge_images(squared_B, squared_B2A, squared_B2A2B)

        checkpoint_step = int(os.path.basename(checkpoint_path).split('-')[1])
        ImageWrite(domain_A_images, 'domain_A2B', checkpoint_step)
        ImageWrite(domain_B_images, 'domain_B2A', checkpoint_step)

    else:
      if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        if FLAGS.checkpoint_step == -1:
          checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
          checkpoint_path = os.path.join(FLAGS.checkpoint_path, 'model.ckpt-%d' % FLAGS.checkpoint_step)

        if os.path.basename(checkpoint_path) + '.data-00000-of-00001' in os.listdir(FLAGS.checkpoint_path):
          print(os.path.basename(checkpoint_path))
        else:
          raise ValueError("No checkpoint file found in: %s" % checkpoint_path)
      else:
        raise ValueError("checkpoint_path must be folder path")
      

      A2B, B2A, A2B2A, B2A2B = run_generator_once(saver, checkpoint_path, model, images_A, images_B)

      squared_A = make_squared_image(images_A)
      squared_B = make_squared_image(images_B)
      squared_A2B = make_squared_image(A2B)
      squared_B2A = make_squared_image(B2A)
      squared_A2B2A = make_squared_image(A2B2A)
      squared_B2A2B = make_squared_image(B2A2B)

      domain_A_images = merge_images(squared_A, squared_A2B, squared_A2B2A)
      domain_B_images = merge_images(squared_B, squared_B2A, squared_B2A2B)

      checkpoint_step = int(os.path.basename(checkpoint_path).split('-')[1])
      ImageWrite(domain_A_images, 'domain_A2B', checkpoint_step)
      ImageWrite(domain_B_images, 'domain_B2A', checkpoint_step)

    print('complete image translate...')


if __name__ == '__main__':
  tf.app.run()
