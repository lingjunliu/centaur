
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_stringupper_inputs():
    list_of_inputs = []

    # Input 1: Scalar lowercase ASCII string
    input_dict = {
        'input': np.array(b"hello world", dtype=np.object_),
        'encoding': '',
        'name': 'scalar_ascii'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor with mixed case ASCII strings
    input_dict = {
        'input': np.array([b"CamelCase", b"all lower", b"ALL UPPER", b"with123"], dtype=np.object_),
        'encoding': '',
        'name': '1d_mixed_ascii'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor with ASCII strings, numbers, and symbols
    input_dict = {
        'input': np.array([[b"aB cD", b"123-456"], [b"!@#$%", b"Should be upper"]], dtype=np.object_),
        'encoding': '',
        'name': '2d_symbols_ascii'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar UTF-8 string with characters that have uppercase equivalents
    input_dict = {
        'input': np.array('ça va bien'.encode('utf-8'), dtype=np.object_),
        'encoding': 'utf-8',
        'name': 'scalar_utf8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D tensor with mixed UTF-8 strings
    input_dict = {
        'input': np.array(['Straße'.encode('utf-8'), '你好'.encode('utf-8'), 'mixedCaseÜ'.encode('utf-8')], dtype=np.object_),
        'encoding': 'utf-8',
        'name': '1d_utf8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor containing an empty string
    input_dict = {
        'input': np.array([b"not empty", b""], dtype=np.object_),
        'encoding': '',
        'name': 'empty_string_ascii'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty 1D tensor (shape=[0])
    input_dict = {
        'input': np.array([], dtype=np.object_),
        'encoding': '',
        'name': 'empty_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D tensor with ASCII strings
    input_dict = {
        'input': np.array([[[b"one", b"two"], [b"three", b"four"]], [[b"five", b"six"], [b"seven", b"eight"]]], dtype=np.object_),
        'encoding': '',
        'name': '3d_ascii'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All uppercase ASCII input, should remain unchanged
    input_dict = {
        'input': np.array(b"ALREADY UPPERCASE", dtype=np.object_),
        'encoding': '',
        'name': 'already_upper'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D UTF-8 tensor with special casing (ß -> SS)
    input_dict = {
        'input': np.array([['guten tag'.encode('utf-8'), 'au revoir'.encode('utf-8')], ['спасибо'.encode('utf-8'), 'ß'.encode('utf-8')]], dtype=np.object_),
        'encoding': 'utf-8',
        'name': '2d_utf8_special_casing'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Tensor with only non-alphabetic ASCII characters
    input_dict = {
        'input': np.array(b"12345 !@#$%^&*()", dtype=np.object_),
        'encoding': '',
        'name': 'non_alpha_ascii'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Empty 2D tensor (shape=[2,0])
    input_dict = {
        'input': np.empty(shape=(2,0), dtype=np.object_),
        'encoding': 'utf-8',
        'name': 'empty_2d_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.StringUpper"] = tf_raw_ops_stringupper_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.StringUpper' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringUpper'.")

check_valid('tf.raw_ops.StringUpper', generated_inputs['tf.raw_ops.StringUpper'], lib="tf", suffix=0)
