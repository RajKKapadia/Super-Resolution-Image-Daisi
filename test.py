from PIL import Image

from main import generate_super_resolution_image

# Read the input file using PIL.Image
IMAGE_PATH = 'input\\40157448-eff91f06-5953-11e8-9a37-f6b5693fa03f.png'

image = Image.open(IMAGE_PATH)

image = generate_super_resolution_image(image=image)

image.show()