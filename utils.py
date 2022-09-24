import os

import tensorflow as tf
from PIL import Image
import numpy as np

import config


def preprocess_image(image_path: str) -> tf.Tensor:
  ''' Loads image from path and preprocesses to make it model ready
      
      Parameters:
      - image_path: Path to the image file

      Return:
      - tf.Tensor
  '''
  hr_image = tf.image.decode_image(tf.io.read_file(image_path))
  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)


def downscale_image(image: tf.Tensor, scale: int = 4) -> tf.Tensor:
    ''' Scales down images using bicubic downsampling.
        
        Parameters:
        - image: 3D or 4D tensor of preprocessed image
    '''
    image_size = []
    if len(image.shape) == 3:
        image_size = [image.shape[1], image.shape[0]]
    else:
        raise ValueError('Dimension mismatch. Can work only on single image.')

    image = tf.squeeze(
        tf.cast(
            tf.clip_by_value(image, 0, 255), tf.uint8
        )
    )

    lr_image = np.asarray(
    Image.fromarray(
        image.numpy()
    ).resize(
        [image_size[0] // scale,
        image_size[1] // scale],
        Image.Resampling.BICUBIC
        )
    )

    lr_image = tf.expand_dims(lr_image, 0)
    lr_image = tf.cast(lr_image, tf.float32)
    return lr_image


def save_image(image, filename):
  ''' Saves unscaled Tensor Images.
      
      Parameters:
      - image: 3D image tensor. [height, width, channels]
      - filename: Name of the file to save.
  '''
  if not isinstance(image, Image.Image):
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  image.save(
    os.path.join(
        config.OUTPUT_DIR,
        f'{filename}'
    )
  )
