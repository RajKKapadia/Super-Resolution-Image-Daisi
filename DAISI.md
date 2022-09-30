# Super Resolution Image

This Daisi is a simple application that generates a super-resolution image from a low-resolution image using [ESRGAN](https://arxiv.org/pdf/1809.00219.pdf). I have leveraged the facility of [Tensorflow Hub](https://tfhub.dev/) and this awesome blog from [Tensorflow](https://www.tensorflow.org/hub/tutorials/image_enhancing) to develop this application. This is just an implementation of the existing work.

The technology I have used are:
* [Tensorflow Hub](https://tfhub.dev/)
* [Tensorflow blog](https://www.tensorflow.org/hub/tutorials/image_enhancing)
* [Streamlit](https://streamlit.io/)

```python
import pydaisi as pyd
from PIL import Image

image = Image.open(YOUR_IMAGE_PATH)

super_resolution_image = pyd.Daisi('rajkkapadia/Super Resolution Image')
result = super_resolution_image.generate_super_resolution_image(image=image).value

print(result)
```
The result of the function call will a Numpy array, you can use it and convert it in to an image using the following piece of code.

```python
from PIL import Image
image = Image.fromarray(result)
image.show()
```