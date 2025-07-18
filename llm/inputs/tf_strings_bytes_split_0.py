
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_bytes_split_inputs():
    """
    Generates a list of valid inputs for the tf.strings.bytes_split function.
    """
    list_of_inputs = []

    # Input 1: Simple scalar string
    input_dict = {
        'input': np.array('hello', dtype=object),
        'name': 'scalar_split'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor of simple ASCII strings
    input_dict = {
        'input': np.array(['hello', 'world'], dtype=object),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D tensor including an empty string
    input_dict = {
        'input': np.array(['abc', '', 'def'], dtype=object),
        'name': 'with_empty_string'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensor with multi-byte unicode characters
    input_dict = {
        'input': np.array(['你好', 'world', 'こんにちは'], dtype=object),
        'name': 'unicode_split'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D tensor of strings
    input_dict = {
        'input': np.array([['a', 'b'], ['c', 'd']], dtype=object),
        'name': '2d_split'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D tensor with varying length strings
    input_dict = {
        'input': np.array([['short', 'a much longer string'], ['c', 'd']], dtype=object),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor of strings
    input_dict = {
        'input': np.array([[['a'], ['b']], [['c'], ['d']]], dtype=object),
        'name': '3d_split'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with emojis and other non-ASCII characters
    input_dict = {
        'input': np.array(['TF ❤️', '你好', '🎉'], dtype=object),
        'name': 'emoji_split'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with numeric strings
    input_dict = {
        'input': np.array(['123', '4567', '890'], dtype=object),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with special characters
    input_dict = {
        'input': np.array(['!@#$', '%^&*()', '`~-=_+'], dtype=object),
        'name': 'special_chars_split'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: An empty 1D tensor
    input_dict = {
        'input': np.array([], dtype=object),
        'name': 'empty_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: A 1D tensor containing only a single empty string
    input_dict = {
        'input': np.array([''], dtype=object),
        'name': 'single_empty_string_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.strings.bytes_split"] = tf_strings_bytes_split_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.strings.bytes_split' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.bytes_split'.")

check_valid('tf.strings.bytes_split', generated_inputs['tf.strings.bytes_split'], lib="tf", suffix=0)
