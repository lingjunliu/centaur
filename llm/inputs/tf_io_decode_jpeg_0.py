
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_jpeg_inputs():
    list_of_inputs = []

    # Input 1
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x03\x02\x02\x02\x02\x02\x03\x02\x02\x02\x03\x03\x03\x03\x04\x06\x04\x04\x04\x04\x04\x08\x06\x06\x05\x06\t\x08\n\n\t\x08\t\t\n\x0c\x08\x08\x0b\n\x0c\x0e\x0e\x0b\n\x0c\x11\x0e\x0f\x10\x0e\x11\x11\x13\x16\x13\x11\x12\x11\x11\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xc2\x3f\x0b\xff\xd9')
    channels = 0
    ratio = 1
    fancy_upscaling = True
    try_recover_truncated = False
    acceptable_fraction = 1.0
    dct_method = ""
    name = None
    input_dict = {
        "contents": contents,
        "channels": channels,
        "ratio": ratio,
        "fancy_upscaling": fancy_upscaling,
        "try_recover_truncated": try_recover_truncated,
        "acceptable_fraction": acceptable_fraction,
        "dct_method": dct_method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x03\x02\x02\x02\x02\x02\x03\x02\x02\x02\x03\x03\x03\x03\x04\x06\x04\x04\x04\x04\x04\x08\x06\x06\x05\x06\t\x08\n\n\t\x08\t\t\n\x0c\x08\x08\x0b\n\x0c\x0e\x0e\x0b\n\x0c\x11\x0e\x0f\x10\x0e\x11\x11\x13\x16\x13\x11\x12\x11\x11\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xc2\x3f\x0b\xff\xd9')
    channels = 1
    ratio = 2
    fancy_upscaling = False
    try_recover_truncated = True
    acceptable_fraction = 0.5
    dct_method = "INTEGER_FAST"
    name = "decode_jpeg_op"
    input_dict = {
        "contents": contents,
        "channels": channels,
        "ratio": ratio,
        "fancy_upscaling": fancy_upscaling,
        "try_recover_truncated": try_recover_truncated,
        "acceptable_fraction": acceptable_fraction,
        "dct_method": dct_method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x03\x02\x02\x02\x02\x02\x03\x02\x02\x02\x03\x03\x03\x03\x04\x06\x04\x04\x04\x04\x04\x08\x06\x06\x05\x06\t\x08\n\n\t\x08\t\t\n\x0c\x08\x08\x0b\n\x0c\x0e\x0e\x0b\n\x0c\x11\x0e\x0f\x10\x0e\x11\x11\x13\x16\x13\x11\x12\x11\x11\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xc2\x3f\x0b\xff\xd9')
    channels = 3
    ratio = 1
    fancy_upscaling = True
    try_recover_truncated = False
    acceptable_fraction = 0.9
    dct_method = ""
    name = "another_name"
    input_dict = {
        "contents": contents,
        "channels": channels,
        "ratio": ratio,
        "fancy_upscaling": fancy_upscaling,
        "try_recover_truncated": try_recover_truncated,
        "acceptable_fraction": acceptable_fraction,
        "dct_method": dct_method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x03\x02\x02\x02\x02\x02\x03\x02\x02\x02\x03\x03\x03\x03\x04\x06\x04\x04\x04\x04\x04\x08\x06\x06\x05\x06\t\x08\n\n\t\x08\t\t\n\x0c\x08\x08\x0b\n\x0c\x0e\x0e\x0b\n\x0c\x11\x0e\x0f\x10\x0e\x11\x11\x13\x16\x13\x11\x12\x11\x11\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xc2\x3f\x0b\xff\xd9')
    channels = 0
    ratio = 4
    fancy_upscaling = False
    try_recover_truncated = True
    acceptable_fraction = 0.01
    dct_method = ""
    name = ""
    input_dict = {
        "contents": contents,
        "channels": channels,
        "ratio": ratio,
        "fancy_upscaling": fancy_upscaling,
        "try_recover_truncated": try_recover_truncated,
        "acceptable_fraction": acceptable_fraction,
        "dct_method": dct_method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x03\x02\x02\x02\x02\x02\x03\x02\x02\x02\x03\x03\x03\x03\x04\x06\x04\x04\x04\x04\x04\x08\x06\x06\x05\x06\t\x08\n\n\t\x08\t\t\n\x0c\x08\x08\x0b\n\x0c\x0e\x0e\x0b\n\x0c\x11\x0e\x0f\x10\x0e\x11\x11\x13\x16\x13\x11\x12\x11\x11\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xc2\x3f\x0b\xff\xd9')
    channels = 1
    ratio = 1
    fancy_upscaling = True
    try_recover_truncated = False
    acceptable_fraction = 1.0
    dct_method = ""
    name = "test_name"
    input_dict = {
        "contents": contents,
        "channels": channels,
        "ratio": ratio,
        "fancy_upscaling": fancy_upscaling,
        "try_recover_truncated": try_recover_truncated,
        "acceptable_fraction": acceptable_fraction,
        "dct_method": dct_method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x03\x02\x02\x02\x02\x02\x03\x02\x02\x02\x03\x03\x03\x03\x04\x06\x04\x04\x04\x04\x04\x08\x06\x06\x05\x06\t\x08\n\n\t\x08\t\t\n\x0c\x08\x08\x0b\n\x0c\x0e\x0e\x0b\n\x0c\x11\x0e\x0f\x10\x0e\x11\x11\x13\x16\x13\x11\x12\x11\x11\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xc2\x3f\x0b\xff\xd9')
    channels = 3
    ratio = 2
    fancy_upscaling = False
    try_recover_truncated = True
    acceptable_fraction = 0.75
    dct_method = "INTEGER_FAST"
    name = "jpeg_decoder"
    input_dict = {
        "contents": contents,
        "channels": channels,
        "ratio": ratio,
        "fancy_upscaling": fancy_upscaling,
        "try_recover_truncated": try_recover_truncated,
        "acceptable_fraction": acceptable_fraction,
        "dct_method": dct_method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x03\x02\x02\x02\x02\x02\x03\x02\x02\x02\x03\x03\x03\x03\x04\x06\x04\x04\x04\x04\x04\x08\x06\x06\x05\x06\t\x08\n\n\t\x08\t\t\n\x0c\x08\x08\x0b\n\x0c\x0e\x0e\x0b\n\x0c\x11\x0e\x0f\x10\x0e\x11\x11\x13\x16\x13\x11\x12\x11\x11\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xc2\x3f\x0b\xff\xd9')
    channels = 0
    ratio = 1
    fancy_upscaling = True
    try_recover_truncated = True
    acceptable_fraction = 0.2
    dct_method = "INTEGER_ACCURATE"
    name = "truncated_test"
    input_dict = {
        "contents": contents,
        "channels": channels,
        "ratio": ratio,
        "fancy_upscaling": fancy_upscaling,
        "try_recover_truncated": try_recover_truncated,
        "acceptable_fraction": acceptable_fraction,
        "dct_method": dct_method,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    contents = np.array(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x03\x02\x02\x02\x02\x02\x03\x02\x02\x02\x03\x03\x03\x03\x04\x06\x04\x04\x04\x04\x04\x08\x06\x06\x05\x06\t\x08\n\n\t\x08\t\t\n\x0c\x08\x08\x0b\n\x0c\x0e\x0e\x0b\n\x0c\x11\x0e\x0f\x10\x0e\x11\x11\x13\x16\x13\x11\x12\x11\x11\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xc2\x3f\x0b')

    channels = 0
    ratio = 1
    fancy_upscaling = True
    try_recover_truncated = True
    acceptable_fraction = 0.9
    dct_method = ""
    name = "truncated_image"
    input_dict = {
        "contents": contents,
        "channels": channels,
        "ratio": ratio,
        "fancy_upscaling": fancy_upscaling,
        "try_recover_truncated": try_recover_truncated,
        "acceptable_fraction": acceptable_fraction,
        "dct_method": dct_method,
        "name": name
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
