import uuid
import os
import time
from pathlib import Path

from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st

import config
import utils

model = hub.load(config.SAVED_MODEL_PATH)

def generate_super_resolution_image(image_path: str) -> np.array:
    ''' 
    This function will generate a super resolution image provided a valid PNG, JPEG, JPG image path. 
    This will return a numpy array.
        
    :param image_path(str): Path of the image file
    
    :return np.array: Result will be a numpy array
    Return:
    -------
    - np.array
    '''
    file_extension = Path(image_path).suffix
    if file_extension not in ['.png', '.jpg', '.jpeg']:
        return 'Please pass only .png, .jpg, and .jpeg files.'
    hr_image = utils.preprocess_image(image_path=image_path)
    lr_image = utils.downscale_image(tf.squeeze(hr_image), scale=1)
    fake_image = model(lr_image)
    fake_image = tf.squeeze(fake_image)
    fake_image_path = os.path.join(
        config.OUTPUT_DIR,
        f'{uuid.uuid4()}.jpg'
    )
    utils.save_image(fake_image, fake_image_path)

    image = Image.open(fp=fake_image_path)

    image = np.array(image)

    os.unlink(fake_image_path)

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
        original_image_path = os.path.join(
            config.INPUT_DIR,
            f'{uuid.uuid4()}.{uploaded_image.name.split(".")[1]}'
        )
        with open(original_image_path, 'wb') as f:
            f.write(uploaded_image.getbuffer())
        super_image = generate_super_resolution_image(original_image_path)
        with col2:
            col2.image(
                image=super_image,
                caption='Enhanced image',
                width=300
            )
        fake_image = Image.fromarray(
            obj=super_image
        )
        fake_image_path = os.path.join(
            config.OUTPUT_DIR,
            f'{uuid.uuid4()}.jpg'
        )
        fake_image.save(
            fp=fake_image_path
        )
        with open(fake_image_path, 'rb') as f:
            col2.download_button(
                label='Download image',
                data = f,
                file_name=os.path.basename(fake_image_path)
            )

if __name__ == '__main__':
    st_ui()