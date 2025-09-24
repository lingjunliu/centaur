
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_strings_join_inputs():
    """
    Generates a list of valid inputs for the tf.strings.join function.
    The 'inputs' parameter is a single tensor to accommodate execution environments
    that expect an object with a .shape attribute, invoking the reduction
    behavior of tf.strings.join.
    """
    list_of_inputs = []

    # Input 1: Basic 2D array, default separator. Joins along axis=-1.
    input_dict_1 = {
        'inputs': np.array([['a', 'b', 'c'], ['d', 'e', 'f']], dtype=object),
        'separator': '',
        'name': 'reduce_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D array with a space separator.
    input_dict_2 = {
        'inputs': np.array([['hello', 'world'], ['goodbye', 'moon']], dtype=object),
        'separator': ' ',
        'name': 'reduce_2d_space'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D array with a dash separator.
    input_dict_3 = {
        'inputs': np.array([[['a', 'b'], ['c', 'd']], [['e', 'f'], ['g', 'h']]], dtype=object),
        'separator': '-',
        'name': 'reduce_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 1D array. Reduction results in a scalar.
    input_dict_4 = {
        'inputs': np.array(['one', 'two', 'three'], dtype=object),
        'separator': ',',
        'name': 'reduce_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Array with empty strings.
    input_dict_5 = {
        'inputs': np.array([['', 'non-empty'], ['another', '']], dtype=object),
        'separator': '*',
        'name': 'reduce_empty_strings'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Array with a dimension of size 1.
    input_dict_6 = {
        'inputs': np.array([['single1'], ['single2']], dtype=object),
        'separator': ' ',
        'name': 'reduce_dim_one'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty array with rank 2.
    input_dict_7 = {
        'inputs': np.empty(shape=(2, 0), dtype=object),
        'separator': 'X',
        'name': 'reduce_empty_array'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Unicode characters.
    input_dict_8 = {
        'inputs': np.array([['你好', '世界'], ['👋', '🌍']], dtype=object),
        'separator': ' ',
        'name': 'reduce_unicode'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Using a multi-character separator.
    input_dict_9 = {
        'inputs': np.array([['path', 'to'], ['file', 'name']], dtype=object),
        'separator': '//',
        'name': 'reduce_multi_char_separator'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Array with only one column. Result should be the column itself without separators.
    input_dict_10 = {
        'inputs': np.array([['a'], ['b'], ['c']], dtype=object),
        'separator': '-',
        'name': 'reduce_one_column'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Array with only one row. Result should be a scalar string.
    input_dict_11 = {
        'inputs': np.array([['a', 'b', 'c']], dtype=object),
        'separator': ':',
        'name': 'reduce_one_row'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.strings.join"] = tf_strings_join_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.strings.join' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.join'.")

check_valid('tf.strings.join', generated_inputs['tf.strings.join'], lib="tf", suffix=0)
