
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_strings_upper_inputs():
    list_of_inputs = []

    # Input 1: Basic scalar input
    input_dict = {
        'input': np.array('hello world', dtype=object),
        'encoding': '',
        'name': 'test_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor of strings
    input_dict = {
        'input': np.array(['camelcase', 'string', 'and', 'all caps'], dtype=object),
        'encoding': '',
        'name': 'test_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor of strings
    input_dict = {
        'input': np.array([['a', 'b'], ['c', 'd']], dtype=object),
        'encoding': '',
        'name': 'test_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mixed case strings
    input_dict = {
        'input': np.array(['MiXeD', 'CaSe', 'InPuT'], dtype=object),
        'encoding': '',
        'name': 'test_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Strings with numbers and symbols
    input_dict = {
        'input': np.array('string with 123 and !@#$', dtype=object),
        'encoding': '',
        'name': 'test_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty string scalar
    input_dict = {
        'input': np.array('', dtype=object),
        'encoding': '',
        'name': 'test_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor containing an empty string
    input_dict = {
        'input': np.array(['hello', '', 'world'], dtype=object),
        'encoding': '',
        'name': 'test_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: UTF-8 encoding with non-ASCII characters
    input_dict = {
        'input': np.array(['français', 'español', 'über'], dtype=object),
        'encoding': 'utf-8',
        'name': 'test_utf8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High-dimensional tensor (3D)
    input_dict = {
        'input': np.array([[['a', 'b'], ['c', 'd']], [['e', 'f'], ['g', 'h']]], dtype=object),
        'encoding': '',
        'name': 'test_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Input already in uppercase
    input_dict = {
        'input': np.array('ALREADY UPPER', dtype=object),
        'encoding': '',
        'name': 'test_upper'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Scalar UTF-8 input
    input_dict = {
        'input': np.array('你好世界', dtype=object),
        'encoding': 'utf-8',
        'name': 'test_chinese'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: No name parameter
    input_dict = {
        'input': np.array(['no', 'name'], dtype=object),
        'encoding': '',
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs["tf.strings.upper"] = tf_strings_upper_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.strings.upper' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.upper'.")

check_valid('tf.strings.upper', generated_inputs['tf.strings.upper'], lib="tf", suffix=0)
