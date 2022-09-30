from PIL import Image
import numpy as np

from main import generate_super_resolution_image

# Read the input file using PIL.Image
IMAGE_PATH = 'input\\black_and_white.jpg'

image = Image.open(IMAGE_PATH)

image_array = generate_super_resolution_image(image=image)

image = Image.fromarray(obj=image_array)

image.show()