
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_gif_inputs():
    list_of_inputs = []

    # Input 1: Basic valid GIF data
    gif_data = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    input_dict = {"contents": gif_data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Another valid GIF data (slightly different header)
    gif_data = b"GIF87a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    input_dict = {"contents": gif_data, "name": "gif_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Short GIF data
    gif_data = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00"
    input_dict = {"contents": gif_data, "name": "gif_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Minimal GIF data
    gif_data = b"GIF89a"
    input_dict = {"contents": gif_data, "name": "gif_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty GIF data
    gif_data = b""
    input_dict = {"contents": gif_data, "name": "gif_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.decode_gif"] = tf_io_decode_gif_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.decode_gif' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_gif'.")

check_valid('tf.io.decode_gif', generated_inputs['tf.io.decode_gif'], lib="tf", suffix=0)
