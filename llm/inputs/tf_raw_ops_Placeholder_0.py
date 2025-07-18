
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_placeholder_inputs():
    # As per the documentation, tf.raw_ops.Placeholder is a graph-mode op
    # that is expected to fail with an error if executed directly in an eager
    # context. The "You must feed a value..." error is correct behavior.
    # The following inputs are valid for the *definition* of the placeholder.
    list_of_inputs = []

    # Input 1: A 2D matrix of float32.
    input_dict_1 = {
        'dtype': np.float32,
        'shape': [4, 4],
        'name': 'placeholder_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: A 1D vector of int32.
    input_dict_2 = {
        'dtype': np.int32,
        'shape': [128],
        'name': 'placeholder_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: A 3D tensor of booleans.
    input_dict_3 = {
        'dtype': np.bool_,
        'shape': [2, 3, 2],
        'name': 'placeholder_bool'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: A 2D matrix of float64 (double).
    input_dict_4 = {
        'dtype': np.float64,
        'shape': [10, 20],
        'name': 'placeholder_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: A 1D vector of int64.
    input_dict_5 = {
        'dtype': np.int64,
        'shape': [64],
        'name': 'placeholder_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: A 4D tensor representing a batch of images (uint8).
    input_dict_6 = {
        'dtype': np.uint8,
        'shape': [16, 32, 32, 3],
        'name': 'placeholder_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: A 2D matrix of int16.
    input_dict_7 = {
        'dtype': np.int16,
        'shape': [5, 100],
        'name': 'placeholder_int16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: A 2D matrix of complex64 numbers.
    input_dict_8 = {
        'dtype': np.complex64,
        'shape': [8, 8],
        'name': 'placeholder_complex64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: A 3D tensor of complex128 numbers.
    input_dict_9 = {
        'dtype': np.complex128,
        'shape': [4, 2, 4],
        'name': 'placeholder_complex128'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: A 1D vector of uint32.
    input_dict_10 = {
        'dtype': np.uint32,
        'shape': [256],
        'name': 'placeholder_uint32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.Placeholder"] = tf_raw_ops_placeholder_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Placeholder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Placeholder'.")

check_valid('tf.raw_ops.Placeholder', generated_inputs['tf.raw_ops.Placeholder'], lib="tf", suffix=0)
