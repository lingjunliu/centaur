
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_bmp_inputs():
    """
    Generates a list of valid inputs for the tf.io.decode_bmp function.
    """
    # Using pre-generated, known-good BMP byte strings to ensure validity.
    # These are 24-bit, bottom-up, BI_RGB bitmaps with a 54-byte header.
    # The structure and sizes have been meticulously verified to match the BMP specification
    # and what the TensorFlow kernel expects, to avoid the size mismatch error.

    # 1x1 Red BMP (58 bytes total)
    # file_size=58, offset=54, width=1, height=1, bpp=24, image_size=4, row_stride=4
    bmp_1x1_red_bytes = (
        b'BM\x3a\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00'
        b'\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00'
        b'\x01\x00\x18\x00\x00\x00\x00\x00\x04\x00\x00\x00'
        b'\x13\x0b\x00\x00\x13\x0b\x00\x00\x00\x00\x00\x00'
        b'\x00\x00\x00\x00'
        b'\x00\x00\xff\x00'
    )

    # 2x2 Green BMP (70 bytes total)
    # file_size=70, offset=54, width=2, height=2, bpp=24, image_size=16, row_stride=8
    bmp_2x2_green_bytes = (
        b'BM\x46\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00'
        b'\x28\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00'
        b'\x01\x00\x18\x00\x00\x00\x00\x00\x10\x00\x00\x00'
        b'\x13\x0b\x00\x00\x13\x0b\x00\x00\x00\x00\x00\x00'
        b'\x00\x00\x00\x00'
        b'\x00\xff\x00\x00\xff\x00\x00\x00'
        b'\x00\xff\x00\x00\xff\x00\x00\x00'
    )
    
    list_of_inputs = []

    contents_red = np.array(bmp_1x1_red_bytes)
    contents_green = np.array(bmp_2x2_green_bytes)

    # Input 1: Default channels (0), use channels from BMP (3).
    list_of_inputs.append({'contents': contents_red, 'channels': 0, 'name': 'test1'})

    # Input 2: Force 3 channels (RGB).
    list_of_inputs.append({'contents': contents_red, 'channels': 3, 'name': 'test2'})

    # Input 3: Force 4 channels (RGBA), alpha channel will be added.
    list_of_inputs.append({'contents': contents_red, 'channels': 4, 'name': 'test3'})

    # Input 4: Different image (green 2x2), no name, default channels.
    list_of_inputs.append(copy.deepcopy({'contents': contents_green, 'channels': 0}))

    # Input 5: Green image, force 3 channels.
    list_of_inputs.append({'contents': contents_green, 'channels': 3, 'name': 'test4'})

    # Input 6: Green image, force 4 channels, no name.
    list_of_inputs.append(copy.deepcopy({'contents': contents_green, 'channels': 4}))

    # Input 7: Red image with different name.
    list_of_inputs.append({'contents': contents_red, 'channels': 0, 'name': 'test5'})

    # Input 8: Green image with another name.
    list_of_inputs.append({'contents': contents_green, 'channels': 0, 'name': 'test6'})
    
    # Input 9: Red image, 3 channels, no name.
    list_of_inputs.append(copy.deepcopy({'contents': contents_red, 'channels': 3}))
    
    # Input 10: Green image, 4 channels, different name.
    list_of_inputs.append({'contents': contents_green, 'channels': 4, 'name': 'test7'})

    return list_of_inputs

generated_inputs["tf.io.decode_bmp"] = tf_io_decode_bmp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.decode_bmp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_bmp'.")

check_valid('tf.io.decode_bmp', generated_inputs['tf.io.decode_bmp'], lib="tf", suffix=0)
