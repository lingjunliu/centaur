
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os

def tf_io_decode_jpeg_inputs():
    list_of_inputs = []

    # Create a dummy JPEG file if it doesn't exist
    if not os.path.exists('image.jpg'):
        with open('image.jpg', 'wb') as f:
            f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ", #\'.\x02\t\x0b\x08\x0b\x1d!\x1c!4+;\x1e4\x00\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x1c\x00\x00\x00\x08\x00\x00\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2\xcf \x00\xff\xd9')

    # Input 1: Basic valid JPEG data
    try:
        jpeg_data = tf.io.read_file('image.jpg')  # Replace 'image.jpg' with a valid JPEG file
    except tf.errors.NotFoundError:
        print("Error: image.jpg not found. Please provide a valid JPEG file.")
        return []

    input_dict = {
        "contents": jpeg_data,
        "channels": 0,
        "ratio": 1,
        "fancy_upscaling": True,
        "try_recover_truncated": False,
        "acceptable_fraction": 1.0,
        "dct_method": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Grayscale, channels = 1
    input_dict = {
        "contents": jpeg_data,
        "channels": 1,
        "ratio": 1,
        "fancy_upscaling": True,
        "try_recover_truncated": False,
        "acceptable_fraction": 1.0,
        "dct_method": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: RGB, channels = 3
    input_dict = {
        "contents": jpeg_data,
        "channels": 3,
        "ratio": 1,
        "fancy_upscaling": True,
        "try_recover_truncated": False,
        "acceptable_fraction": 1.0,
        "dct_method": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Downscaling ratio = 2
    input_dict = {
        "contents": jpeg_data,
        "channels": 0,
        "ratio": 2,
        "fancy_upscaling": True,
        "try_recover_truncated": False,
        "acceptable_fraction": 1.0,
        "dct_method": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Fancy upscaling = False
    input_dict = {
        "contents": jpeg_data,
        "channels": 0,
        "ratio": 1,
        "fancy_upscaling": False,
        "try_recover_truncated": False,
        "acceptable_fraction": 1.0,
        "dct_method": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Try Recover Truncated = True, Acceptable Fraction = 0.5
    # Create a truncated JPEG file
    try:
        with open('image.jpg', 'rb') as f:  # Replace 'image.jpg' with a valid JPEG file
            truncated_jpeg_data = f.read(100) # Read only first 100 bytes
    except FileNotFoundError:
        print("Error: image.jpg not found. Please provide a valid JPEG file.")
        return []

    input_dict = {
        "contents": tf.convert_to_tensor(truncated_jpeg_data, dtype=tf.string),
        "channels": 0,
        "ratio": 1,
        "fancy_upscaling": True,
        "try_recover_truncated": True,
        "acceptable_fraction": 0.5,
        "dct_method": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: DCT Method = INTEGER_FAST
    input_dict = {
        "contents": jpeg_data,
        "channels": 0,
        "ratio": 1,
        "fancy_upscaling": True,
        "try_recover_truncated": False,
        "acceptable_fraction": 1.0,
        "dct_method": "INTEGER_FAST",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: DCT Method = INTEGER_ACCURATE
    input_dict = {
        "contents": jpeg_data,
        "channels": 0,
        "ratio": 1,
        "fancy_upscaling": True,
        "try_recover_truncated": False,
        "acceptable_fraction": 1.0,
        "dct_method": "INTEGER_ACCURATE",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Name specified
    input_dict = {
        "contents": jpeg_data,
        "channels": 0,
        "ratio": 1,
        "fancy_upscaling": True,
        "try_recover_truncated": False,
        "acceptable_fraction": 1.0,
        "dct_method": "",
        "name": "decode_jpeg_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Ratio = 4
    input_dict = {
        "contents": jpeg_data,
        "channels": 0,
        "ratio": 4,
        "fancy_upscaling": True,
        "try_recover_truncated": False,
        "acceptable_fraction": 1.0,
        "dct_method": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 11: Acceptable Fraction = 0.9
    input_dict = {
        "contents": tf.convert_to_tensor(truncated_jpeg_data, dtype=tf.string),
        "channels": 0,
        "ratio": 1,
        "fancy_upscaling": True,
        "try_recover_truncated": True,
        "acceptable_fraction": 0.9,
        "dct_method": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Ratio = 8
    input_dict = {
        "contents": jpeg_data,
        "channels": 0,
        "ratio": 8,
        "fancy_upscaling": True,
        "try_recover_truncated": False,
        "acceptable_fraction": 1.0,
        "dct_method": "",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.decode_jpeg"] = tf_io_decode_jpeg_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.decode_jpeg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_jpeg'.")

check_valid('tf.io.decode_jpeg', generated_inputs['tf.io.decode_jpeg'], lib="tf", suffix=0)
