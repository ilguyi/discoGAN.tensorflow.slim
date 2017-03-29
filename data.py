"""
  Almost codes related to read data are borrowed from SKTBrain official source codes
  https://github.com/SKTBrain/DiscoGAN
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import pandas as pd
import copy

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


dataset_path = './datasets/'
celebA_path = os.path.join(dataset_path, 'celebA')


def _read_attr_file( attr_path, image_dir ):
    f = open( attr_path )
    lines = f.readlines()
    lines = map(lambda line: line.strip(), lines)
    columns = ['image_path'] + lines[1].split()
    lines = lines[2:]

    items = map(lambda line: line.split(), lines)
    df = pd.DataFrame( items, columns=columns )
    df['image_path'] = df['image_path'].map( lambda x: os.path.join( image_dir, x ) )

    return df


def _get_celebA_files(style_A, style_B, constraint, constraint_type, test=False, n_test=200):
    attr_file = os.path.join( celebA_path, 'list_attr_celeba.txt' )
    image_dir = os.path.join( celebA_path, 'img_align_celeba' )
    image_data = _read_attr_file( attr_file, image_dir )

    if constraint:
        image_data = image_data[ image_data[constraint] == constraint_type]

    style_A_data = image_data[ image_data[style_A] == '1']['image_path'].values
    if style_B:
        style_B_data = image_data[ image_data[style_B] == '1']['image_path'].values
    else:
        style_B_data = image_data[ image_data[style_A] == '-1']['image_path'].values

    if test == False:
        return style_A_data[:-n_test], style_B_data[:-n_test]
    if test == True:
        return style_A_data[-n_test:], style_B_data[-n_test:]


def _shuffle_data(da, db):
    a_idx = range(len(da))
    np.random.shuffle( a_idx )

    b_idx = range(len(db))
    np.random.shuffle(b_idx)

    shuffled_da = np.array(da)[ np.array(a_idx) ]
    shuffled_db = np.array(db)[ np.array(b_idx) ]

    return shuffled_da, shuffled_db


def get_data():
  data_A, data_B = _get_celebA_files(style_A=FLAGS.style_A,
                                     style_B=FLAGS.style_B,
                                     constraint=FLAGS.constraint,
                                     constraint_type=FLAGS.constraint_type,
                                     test=FLAGS.is_test,
                                     n_test=FLAGS.n_test)

  data_A, data_B = _shuffle_data(data_A, data_B)

  return data_A, data_B


def get_batch(batch_size, data_A, data_B, pre_epochs, epochs, step, data_size):
  if int(pre_epochs) < int(epochs):
    data_A, data_B = _shuffle_data(data_A, data_B)

  num_batches_per_epoch = int(data_size / batch_size)
  batch_index = int(step % num_batches_per_epoch)

  A_path = data_A[batch_size * batch_index : batch_size * (batch_index+1)]
  B_path = data_B[batch_size * batch_index : batch_size * (batch_index+1)]

  return A_path, B_path


def read_images(filenames, domain=None, image_size=64):

    images = []
    for fn in filenames:
        image = cv2.imread(fn)
        if image is None:
            continue

        if domain == 'A':
            kernel = np.ones((3,3), np.uint8)
            image = image[:, :256, :]
            image = 255. - image
            image = cv2.dilate( image, kernel, iterations=1 )
            image = 255. - image
        elif domain == 'B':
            image = image[:, 256:, :]

        image = cv2.resize(image, (image_size,image_size))
        image = image.astype(np.float32) / 255.
        # TensorFlow shape (height, width, channels)
        #image = image.transpose(2,0,1)
        images.append( image )

    images = np.stack( images )
    return images






