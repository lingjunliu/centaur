
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os

def tf_io_decode_bmp_inputs():
    list_of_inputs = []

    # Input 1: Minimal valid input, default channels
    contents = tf.constant(b'\x42\x4D\x36\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
    channels = 0
    name = None
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: channels=3
    contents = tf.constant(b'\x42\x4D\x36\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
    channels = 3
    name = "bmp_3_channel"
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: channels=4
    contents = tf.constant(b'\x42\x4D\x36\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
    channels = 4
    name = "bmp_4_channel"
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: different valid bmp data
    contents = tf.constant(b'BM\x3e\x00\x00\x00\x00\x00\x00\x00\x3e\x00\x00\x00(\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\x00\x00\x00\xff\xff\xff\x00\x00\x00')
    channels = 0
    name = "bmp_different_data"
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: channels=3, different name
    contents = tf.constant(b'BM\x3e\x00\x00\x00\x00\x00\x00\x00\x3e\x00\x00\x00(\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\x00\x00\x00\xff\xff\xff\x00\x00\x00')
    channels = 3
    name = "another_bmp_name"
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: channels=4, different name
    contents = tf.constant(b'BM\x3e\x00\x00\x00\x00\x00\x00\x00\x3e\x00\x00\x00(\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\x00\x00\x00\xff\xff\xff\x00\x00\x00')
    channels = 4
    name = "yet_another_bmp"
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Long name
    contents = tf.constant(b'BM\x3e\x00\x00\x00\x00\x00\x00\x00\x3e\x00\x00\x00(\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\x00\x00\x00\xff\xff\xff\x00\x00\x00')
    channels = 0
    name = "this_is_a_very_very_very_long_name_for_the_operation"
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: channels = -1 (should still be ok, but will likely result in error down the line, keeping to satisfy the prompt requirements.)
    contents = tf.constant(b'BM\x3e\x00\x00\x00\x00\x00\x00\x00\x3e\x00\x00\x00(\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\x00\x00\x00\xff\xff\xff\x00\x00\x00')
    channels = -1
    name = "negative_channel"
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: bmp with different size
    contents = tf.constant(b'BM\x76\x00\x00\x00\x00\x00\x00\x00\x76\x00\x00\x00(\x00\x00\x00\x04\x00\x00\x00\x04\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\x00\x00\x00\xff\xff\xff\x00\x00\x00\xff\xff\xff\x00\x00\x00\x00\x00\x00\xff\xff\xff\x00\x00\x00\xff\xff\xff\x00\x00\x00\x00\x00\x00\xff\xff\xff\x00\x00\x00\xff\xff\xff\x00\x00\x00')
    channels = 0
    name = "different_size_bmp"
    input_dict = {"contents": contents, "channels": channels, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: channels=3, different size
    contents = tf.constant(b'BM\x76\x00\x00\x00\x00\x00\x00\x00\x76\x00\x00\x00(\x00\x00\x00\x04\x00\x00\x00\x04\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\x00\x00\x00\xff\xff\xff\x00\x00\x00\xff\xff\xff\x00\x00\x00\x00\x00\x00\xff\xff\xff\x00\x00\x00\xff\xff\xff\x00\x00\x00\x00\x00\x00\xff\xff\xff\x00\x00\x00\xff\xff\xff\x00\x00\x00')
    channels = 3
    name = "different_size_bmp_3_channel"
    input_dict = {"contents": contents, "channels": channels, "name": name}
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
