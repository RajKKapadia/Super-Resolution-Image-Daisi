from PIL import Image

from main import generate_super_resolution_image

IMAGE_PATH = 'input\\8ad77e98-580d-47bf-a674-da03c66694db.jpg'

image_array = generate_super_resolution_image(image_path=IMAGE_PATH)

image = Image.fromarray(obj=image_array)

image.show()