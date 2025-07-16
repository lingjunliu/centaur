
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_bmp_inputs():
    list_of_inputs = []

    # Input 1: Minimal valid BMP data, channels=0, full header
    bmp_data_1 = np.array(b'\x42\x4D\x36\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype=np.string_)

    input_dict = {
        "contents": bmp_data_1,
        "channels": 0,
        "name": "bmp_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Minimal valid BMP data, channels=3, full header
    bmp_data_2 = np.array(b'\x42\x4D\x36\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype=np.string_)

    input_dict = {
        "contents": bmp_data_2,
        "channels": 3,
        "name": "bmp_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Minimal valid BMP data, channels=4, full header
    bmp_data_3 = np.array(b'\x42\x4D\x36\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype=np.string_)
    input_dict = {
        "contents": bmp_data_3,
        "channels": 4,
        "name": "bmp_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Slightly larger BMP, channels=0, full header, some data
    bmp_data_4 = np.array(b'\x42\x4D\x42\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype=np.string_)
    input_dict = {
        "contents": bmp_data_4,
        "channels": 0,
        "name": "bmp_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Slightly larger BMP, channels=3, full header, some data
    bmp_data_5 = np.array(b'\x42\x4D\x42\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype=np.string_)
    input_dict = {
        "contents": bmp_data_5,
        "channels": 3,
        "name": "bmp_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Slightly larger BMP, channels=4, full header, some data
    bmp_data_6 = np.array(b'\x42\x4D\x42\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype=np.string_)
    input_dict = {
        "contents": bmp_data_6,
        "channels": 4,
        "name": "bmp_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Different name, full header, some data
    bmp_data_7 = np.array(b'\x42\x4D\x36\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype=np.string_)
    input_dict = {
        "contents": bmp_data_7,
        "channels": 0,
        "name": "different_name"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Channels = 0, Different BMP, complete header and pixel data
    bmp_data_8 = np.array(b'\x42\x4d\x7a\x00\x00\x00\x00\x00\x00\x00\x76\x00\x00\x00\x28\x00\x00\x00\x0a\x00\x00\x00\x0a\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype=np.string_)
    input_dict = {
        "contents": bmp_data_8,
        "channels": 0,
        "name": "bmp_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: channels = 3, different bmp, complete header and pixel data
    bmp_data_9 = np.array(b'\x42\x4d\x7a\x00\x00\x00\x00\x00\x00\x00\x76\x00\x00\x00\x28\x00\x00\x00\x0a\x00\x00\x00\x0a\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype=np.string_)
    input_dict = {
        "contents": bmp_data_9,
        "channels": 3,
        "name": "bmp_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: channels = 4, different bmp, complete header and pixel data
    bmp_data_10 = np.array(b'\x42\x4d\x7a\x00\x00\x00\x00\x00\x00\x00\x76\x00\x00\x00\x28\x00\x00\x00\x0a\x00\x00\x00\x0a\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00', dtype=np.string_)
    input_dict = {
        "contents": bmp_data_10,
        "channels": 4,
        "name": "bmp_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Valid BMP with RGB data
    bmp_data_11 = np.array(b'BMv\x00\x00\x00\x00\x00\x00\x00v\x00\x00\x00(\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\x00', dtype=np.string_)
    input_dict = {
        "contents": bmp_data_11,
        "channels": 0,
        "name": "bmp_11"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.decode_bmp"] = tf_io_decode_bmp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.decode_bmp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_bmp'.")

check_valid('tf.io.decode_bmp', generated_inputs['tf.io.decode_bmp'], lib="tf", suffix=0)
