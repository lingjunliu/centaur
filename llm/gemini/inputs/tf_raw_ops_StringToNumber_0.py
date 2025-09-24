
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_string_to_number_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.StringToNumber function.
    """
    list_of_inputs = []

    # Input 1: Basic float32 conversion
    input_dict_1 = {
        'string_tensor': np.array(["5.0", "3.0", "7.0"], dtype=object),
        'out_type': tf.float32,
        'name': 'basic_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic int32 conversion with negative numbers
    input_dict_2 = {
        'string_tensor': np.array(["-100", "0", "256"], dtype=object),
        'out_type': tf.int32,
        'name': 'basic_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D tensor to float64
    input_dict_3 = {
        'string_tensor': np.array([["1.123", "2.456"], ["-3.789", "4.0"]], dtype=object),
        'out_type': tf.float64,
        'name': '2d_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Large integers to int64
    input_dict_4 = {
        'string_tensor': np.array(["9223372036854775807", "-9223372036854775808"], dtype=object),
        'out_type': tf.int64,
        'name': 'large_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Unsigned integers to uint32
    input_dict_5 = {
        'string_tensor': np.array(["0", "4294967295"], dtype=object),
        'out_type': tf.uint32,
        'name': 'unsigned_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Input 6: Unsigned large integers to uint64
    input_dict_6 = {
        'string_tensor': np.array(["0", "18446744073709551615"], dtype=object),
        'out_type': tf.uint64,
        'name': 'unsigned_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Scalar (0-D) tensor
    input_dict_7 = {
        'string_tensor': np.array("-12345", dtype=object),
        'out_type': tf.int32,
        'name': 'scalar_int'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Scientific notation to float32
    input_dict_8 = {
        'string_tensor': np.array(["1.23e4", "-4.56E-2", "7.0", "+8.9"], dtype=object),
        'out_type': tf.float32,
        'name': 'scientific_notation'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Empty tensor
    input_dict_9 = {
        'string_tensor': np.array([], dtype=object),
        'out_type': tf.float32,
        'name': 'empty_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Higher-dimensional tensor (3D)
    input_dict_10 = {
        'string_tensor': np.array([[['1', '2'], ['3', '4']], [['5', '6'], ['7', '8']]], dtype=object),
        'out_type': tf.int32,
        'name': '3d_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Strings with leading/trailing whitespace
    input_dict_11 = {
        'string_tensor': np.array(["  5.5  ", "\t-3.2\n", " 7 "], dtype=object),
        'out_type': tf.float64,
        'name': 'whitespace_strings'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.StringToNumber"] = get_string_to_number_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.StringToNumber' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringToNumber'.")

check_valid('tf.raw_ops.StringToNumber', generated_inputs['tf.raw_ops.StringToNumber'], lib="tf", suffix=0)
