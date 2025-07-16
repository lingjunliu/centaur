
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_decode_gif_inputs():
    list_of_inputs = []

    # Input 1: Minimal valid GIF data
    input_dict = {
        "contents": np.array(b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;", dtype=np.string_),
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Valid GIF data with a comment, named op
    input_dict = {
        "contents": np.array(b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;", dtype=np.string_),
        "name": "decode_gif_op"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different valid GIF content
    input_dict = {
        "contents": np.array(b'GIF89a\x02\x00\x02\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\n\x00\x01\x00,\x00\x00\x00\x00\x02\x00\x02\x00\x00\x02\x02D\x01\x00;', dtype=np.string_),
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty name
    input_dict = {
        "contents": np.array(b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;", dtype=np.string_),
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5: Another valid GIF
    input_dict = {
        "contents": np.array(b'GIF89a\x0A\x00\x0A\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x0A\x00\x0A\x00\x00\x02\x02D\x01\x00;', dtype=np.string_),
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger valid GIF (still small, but larger than 1x1 or 2x2)
    input_dict = {
        "contents": np.array(b'GIF89a\x10\x00\x10\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x10\x00\x10\x00\x00\x02\x02D\x01\x00;', dtype=np.string_),
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Name with special characters
    input_dict = {
        "contents": np.array(b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;", dtype=np.string_),
        "name": "decode_gif_op!@#$%^&*()"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Name with numbers
    input_dict = {
        "contents": np.array(b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;", dtype=np.string_),
        "name": "decode_gif_op12345"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Longer name
    input_dict = {
        "contents": np.array(b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;", dtype=np.string_),
        "name": "this_is_a_very_long_name_for_the_gif_decode_operation"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Valid GIF with slightly different formatting
    input_dict = {
        "contents": np.array(b'GIF89a\x04\x00\x04\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x04\x00\x04\x00\x00\x02\x02D\x01\x00;', dtype=np.string_),
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DecodeGif' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeGif'.")

check_valid('tf.raw_ops.DecodeGif', generated_inputs['tf.raw_ops.DecodeGif'], lib="tf", suffix=0)
