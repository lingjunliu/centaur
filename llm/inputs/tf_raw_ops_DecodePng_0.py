
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def get_tf_raw_ops_DecodePng_inputs():
    list_of_inputs = []

    # A minimal 1x1 transparent PNG. Original is RGBA.
    png_contents = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    
    # A minimal 1x1 black JPEG.
    jpeg_contents = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00\x43\x00\x03\x02\x02\x02\x02\x02\x03\x02\x02\x02\x03\x03\x03\x03\x04\x06\x04\x04\x04\x04\x04\x08\x06\x06\x05\x06\x09\x08\n\n\t\x08\t\t\n\x0c\x0f\x0c\n\x0b\x0e\x0b\t\t\r\x11\r\x0e\x0f\x10\x10\x11\x10\n\x0c\x12\x13\x12\x10\x13\x0f\x10\x10\x10\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xd2\xff\xd9'

    # A minimal 1x1 transparent GIF.
    gif_contents = b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
    
    # Input 1: PNG, default channels (0), default dtype (uint8)
    input_dict = {
        'contents': np.array(png_contents),
        'channels': 0,
        'dtype': np.uint8,
        'name': 'png_auto_channels_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: PNG, Grayscale channels (1), dtype uint8
    input_dict = {
        'contents': np.array(png_contents),
        'channels': 1,
        'dtype': np.uint8,
        'name': 'png_grayscale_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: PNG, RGB channels (3), dtype uint8
    input_dict = {
        'contents': np.array(png_contents),
        'channels': 3,
        'dtype': np.uint8,
        'name': 'png_rgb_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: PNG, RGBA channels (4), dtype uint8
    input_dict = {
        'contents': np.array(png_contents),
        'channels': 4,
        'dtype': np.uint8,
        'name': 'png_rgba_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: PNG, default channels (0), dtype uint16
    input_dict = {
        'contents': np.array(png_contents),
        'channels': 0,
        'dtype': np.uint16,
        'name': 'png_auto_channels_uint16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: PNG, RGB channels (3), dtype uint16
    input_dict = {
        'contents': np.array(png_contents),
        'channels': 3,
        'dtype': np.uint16,
        'name': 'png_rgb_uint16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: JPEG, auto channels (0), dtype uint8
    input_dict = {
        'contents': np.array(jpeg_contents),
        'channels': 0,
        'dtype': np.uint8,
        'name': 'jpeg_auto_channels_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: JPEG, force RGB (3), dtype uint8
    input_dict = {
        'contents': np.array(jpeg_contents),
        'channels': 3,
        'dtype': np.uint8,
        'name': 'jpeg_rgb_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: GIF, auto channels (0), dtype uint8
    input_dict = {
        'contents': np.array(gif_contents),
        'channels': 0,
        'dtype': np.uint8,
        'name': 'gif_auto_channels_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: GIF, force RGBA (4), dtype uint8
    input_dict = {
        'contents': np.array(gif_contents),
        'channels': 4,
        'dtype': np.uint8,
        'name': 'gif_rgba_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DecodePng"] = get_tf_raw_ops_DecodePng_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DecodePng' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodePng'.")

check_valid('tf.raw_ops.DecodePng', generated_inputs['tf.raw_ops.DecodePng'], lib="tf", suffix=0)
