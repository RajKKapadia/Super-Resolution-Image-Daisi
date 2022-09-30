import os

SAVED_MODEL_PATH = os.path.join(
    'model',
    'esrgan-tf2_1'
)

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
