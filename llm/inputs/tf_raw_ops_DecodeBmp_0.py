
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_decode_bmp_inputs():
    list_of_inputs = []

    # Input 1: Valid BMP content, default channels, no name
    try:
        bmp_data = tf.io.read_file('testdata/bmp/good.bmp').numpy()
    except:
        bmp_data = b'\x42\x4d\x36\x00\x01\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'  # Minimal valid BMP
    bmp_data = np.array(bmp_data)
    input_dict = {"contents": bmp_data, "channels": np.int32(0), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Valid BMP content, RGB channels, with name
    try:
        bmp_data = tf.io.read_file('testdata/bmp/good.bmp').numpy()
    except:
        bmp_data = b'\x42\x4d\x36\x00\x01\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'  # Minimal valid BMP
    bmp_data = np.array(bmp_data)

    input_dict = {"contents": bmp_data, "channels": np.int32(3), "name": "decode_bmp_rgb"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Valid BMP content, RGBA channels, no name
    try:
        bmp_data = tf.io.read_file('testdata/bmp/good.bmp').numpy()
    except:
        bmp_data = b'\x42\x4d\x36\x00\x01\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'  # Minimal valid BMP
    bmp_data = np.array(bmp_data)
    input_dict = {"contents": bmp_data, "channels": np.int32(4), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DecodeBmp"] = tf_raw_ops_decode_bmp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DecodeBmp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeBmp'.")

check_valid('tf.raw_ops.DecodeBmp', generated_inputs['tf.raw_ops.DecodeBmp'], lib="tf", suffix=0)
