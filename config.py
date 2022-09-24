import os

SAVED_MODEL_PATH = 'https://tfhub.dev/captain-pool/esrgan-tf2/1'

CWD = os.getcwd()
OUTPUT_DIR = os.path.join(
    CWD,
    'output'
)
os.makedirs(
    name=OUTPUT_DIR,
    exist_ok=True
)

INPUT_DIR = os.path.join(
    CWD,
    'input'
)
os.makedirs(
    name=INPUT_DIR,
    exist_ok=True
)
