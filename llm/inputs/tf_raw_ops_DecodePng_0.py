
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_decode_png_inputs():
    list_of_inputs = []

    # Input 1
    contents = np.array(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\xfc\xff\xff?\x03\x00\x05\xfa\x02\x9e\xba\x00\x00\x00\x00IEND\xaeB`\x82', dtype=np.string_)
    channels = 0
    dtype = tf.uint8
    name = "decode_png_1"
    input_dict = {"kwargs": {"contents": contents, "channels": channels, "dtype": dtype, "name": name}, "args": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    contents = np.array(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\xfc\xff\xff?\x03\x00\x05\xfa\x02\x9e\xba\x00\x00\x00\x00IEND\xaeB`\x82', dtype=np.string_)
    channels = 1
    dtype = tf.uint8
    name = "decode_png_2"
    input_dict = {"kwargs": {"contents": contents, "channels": channels, "dtype": dtype, "name": name}, "args": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    contents = np.array(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\xfc\xff\xff?\x03\x00\x05\xfa\x02\x9e\xba\x00\x00\x00\x00IEND\xaeB`\x82', dtype=np.string_)
    channels = 3
    dtype = tf.uint8
    name = "decode_png_3"
    input_dict = {"kwargs": {"contents": contents, "channels": channels, "dtype": dtype, "name": name}, "args": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    contents = np.array(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\xfc\xff\xff?\x03\x00\x05\xfa\x02\x9e\xba\x00\x00\x00\x00IEND\xaeB`\x82', dtype=np.string_)
    channels = 4
    dtype = tf.uint8
    name = "decode_png_4"
    input_dict = {"kwargs": {"contents": contents, "channels": channels, "dtype": dtype, "name": name}, "args": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    contents = np.array(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\xfc\xff\xff?\x03\x00\x05\xfa\x02\x9e\xba\x00\x00\x00\x00IEND\xaeB`\x82', dtype=np.string_)
    channels = 0
    dtype = tf.uint16
    name = "decode_png_5"
    input_dict = {"kwargs": {"contents": contents, "channels": channels, "dtype": dtype, "name": name}, "args": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    contents = np.array(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x06\x00\x00\x00\xf4@\xaf\x1a\x00\x00\x00\x19IDAT(\x91\x03\x01\x00\x02\xff\xff\x00\xf1\xfa\x05\x06\x90k\x17\x00\x00\x00\x00IEND\xaeB`\x82', dtype=np.string_)
    channels = 0
    dtype = tf.uint8
    name = "decode_png_6"
    input_dict = {"kwargs": {"contents": contents, "channels": channels, "dtype": dtype, "name": name}, "args": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    contents = np.array(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x06\x00\x00\x00\xf4@\xaf\x1a\x00\x00\x00\x19IDAT(\x91\x03\x01\x00\x02\xff\xff\x00\xf1\xfa\x05\x06\x90k\x17\x00\x00\x00\x00IEND\xaeB`\x82', dtype=np.string_)
    channels = 1
    dtype = tf.uint8
    name = "decode_png_7"
    input_dict = {"kwargs": {"contents": contents, "channels": channels, "dtype": dtype, "name": name}, "args": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    contents = np.array(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x06\x00\x00\x00\xf4@\xaf\x1a\x00\x00\x00\x19IDAT(\x91\x03\x01\x00\x02\xff\xff\x00\xf1\xfa\x05\x06\x90k\x17\x00\x00\x00\x00IEND\xaeB`\x82', dtype=np.string_)
    channels = 3
    dtype = tf.uint8
    name = "decode_png_8"
    input_dict = {"kwargs": {"contents": contents, "channels": channels, "dtype": dtype, "name": name}, "args": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    contents = np.array(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x06\x00\x00\x00\xf4@\xaf\x1a\x00\x00\x00\x19IDAT(\x91\x03\x01\x00\x02\xff\xff\x00\xf1\xfa\x05\x06\x90k\x17\x00\x00\x00\x00IEND\xaeB`\x82', dtype=np.string_)
    channels = 4
    dtype = tf.uint8
    name = "decode_png_9"
    input_dict = {"kwargs": {"contents": contents, "channels": channels, "dtype": dtype, "name": name}, "args": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    contents = np.array(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x06\x00\x00\x00\xf4@\xaf\x1a\x00\x00\x00\x19IDAT(\x91\x03\x01\x00\x02\xff\xff\x00\xf1\xfa\x05\x06\x90k\x17\x00\x00\x00\x00IEND\xaeB`\x82', dtype=np.string_)
    channels = 0
    dtype = tf.uint16
    name = "decode_png_10"
    input_dict = {"kwargs": {"contents": contents, "channels": channels, "dtype": dtype, "name": name}, "args": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DecodePng"] = tf_raw_ops_decode_png_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DecodePng' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodePng'.")

check_valid('tf.raw_ops.DecodePng', generated_inputs['tf.raw_ops.DecodePng'], lib="tf", suffix=0)
