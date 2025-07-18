
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_gif_inputs():
    """
    Generate a list of valid inputs for tf.io.decode_gif.
    """
    list_of_inputs = []

    # Hardcoded minimal, uncompressed image byte strings to avoid external dependencies.
    # 1x1 black GIF
    MINIMAL_GIF_BLACK = b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
    # 1x1 white GIF
    MINIMAL_GIF_WHITE = b'GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
    
    # Case 1: Minimal black GIF, no name
    input_dict_1 = {
        'contents': np.array(MINIMAL_GIF_BLACK),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: Minimal black GIF, with a name
    input_dict_2 = {
        'contents': np.array(MINIMAL_GIF_BLACK),
        'name': 'decode_black_gif'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Minimal white GIF, no name
    input_dict_3 = {
        'contents': np.array(MINIMAL_GIF_WHITE),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Minimal white GIF, with a name
    input_dict_4 = {
        'contents': np.array(MINIMAL_GIF_WHITE),
        'name': 'decode_white_gif'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Black GIF with another name
    input_dict_5 = {
        'contents': np.array(MINIMAL_GIF_BLACK),
        'name': 'black_gif_op_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: White GIF with another name
    input_dict_6 = {
        'contents': np.array(MINIMAL_GIF_WHITE),
        'name': 'white_gif_op_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Case 7: Black GIF with yet another name
    input_dict_7 = {
        'contents': np.array(MINIMAL_GIF_BLACK),
        'name': 'another_op'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Case 8: White GIF with yet another name
    input_dict_8 = {
        'contents': np.array(MINIMAL_GIF_WHITE),
        'name': 'some_name'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: Black GIF, empty string name
    input_dict_9 = {
        'contents': np.array(MINIMAL_GIF_BLACK),
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: White GIF, empty string name
    input_dict_10 = {
        'contents': np.array(MINIMAL_GIF_WHITE),
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.io.decode_gif"] = tf_io_decode_gif_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.decode_gif' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_gif'.")

check_valid('tf.io.decode_gif', generated_inputs['tf.io.decode_gif'], lib="tf", suffix=0)
