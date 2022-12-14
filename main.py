import uuid
import os
os.environ['TFHUB_DOWNLOAD_PROGRESS'] = 'True'

import PIL
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st

import config
import utils

model = hub.load(config.SAVED_MODEL_PATH)

def generate_super_resolution_image(image: PIL.Image) -> np.ndarray:
    '''
    This function will generate a super resolution from a low resolution image, It will take a valid PNG, JPEG, JPG image path as input, and the output will be a numpy array.
        
    :param image(PIL.Image or Numpy.array): an image as either a PIL image, or a Numpy array
    
    :return PIL.Image: Returns a PIL Image objcet, you can use .save to save the image and .show to view the image
    '''
    hr_image = utils.preprocess_image(image=image)
    lr_image = utils.downscale_image(tf.squeeze(hr_image), scale=1)
    fake_image = model(lr_image)
    fake_image = tf.squeeze(fake_image)
    image = tf.keras.preprocessing.image.array_to_img(fake_image)

    return image


def st_ui():
    ''' Function to render the Streamlit UI.
    '''
    st.title('Generate super resolution image')
    with st.container():
        st.write('Enhance your image quality.')
        st.write('This application is developed using information from this [link](https://www.tensorflow.org/hub/tutorials/image_enhancing).')

    st.sidebar.subheader('Upload your image here...')
    uploaded_image = st.sidebar.file_uploader(
        'Upload image',
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=False,
        key=None,
        help='Your low resolution image.'
    )
    button = st.sidebar.button('Enhance image')
    col1, col2 = st.columns(2)
    if button and uploaded_image:
        with col1:
            st.image(
                image=uploaded_image,
                caption='Original image',
                width=300
            )
        original_image = PIL.Image.open(uploaded_image)
        super_image = generate_super_resolution_image(original_image)
        with col2:
            col2.image(
                image=super_image,
                caption='Enhanced image',
                width=300
            )

if __name__ == '__main__':
    st_ui()