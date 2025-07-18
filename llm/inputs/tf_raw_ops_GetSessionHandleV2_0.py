
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_getsessionhandlev2_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 tensor
    input_dict = {
        'value': np.array([1.0, 2.5, -3.0, 4.2], dtype=np.float32),
        'name': 'handle_float_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 tensor with negative values
    input_dict = {
        'value': np.array([[-1, -2, -3], [3, 4, 5]], dtype=np.int32),
        'name': 'handle_int_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D bool tensor
    input_dict = {
        'value': np.array([[[True, False], [False, True]], [[False, False], [True, True]]], dtype=np.bool_),
        'name': 'handle_bool_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D (scalar) int64 tensor
    input_dict = {
        'value': np.array(42, dtype=np.int64),
        'name': 'handle_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float64 tensor
    input_dict = {
        'value': np.random.rand(3, 2).astype(np.float64),
        'name': 'handle_float64_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float16 tensor
    input_dict = {
        'value': np.array([1.1, 2.2, 3.3], dtype=np.float16),
        'name': 'handle_float16_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 2D uint8 tensor
    input_dict = {
        'value': np.array([[0, 128, 255], [1, 127, 254]], dtype=np.uint8),
        'name': 'handle_uint8_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High-rank (5D) int8 tensor
    input_dict = {
        'value': np.ones((1, 2, 1, 2, 1), dtype=np.int8),
        'name': 'handle_int8_5d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with all zeros
    input_dict = {
        'value': np.zeros((2, 5), dtype=np.int16),
        'name': 'handle_zeros_int16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: A larger 1D tensor
    input_dict = {
        'value': np.arange(50, dtype=np.float32),
        'name': 'handle_large_float_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.GetSessionHandleV2"] = tf_raw_ops_getsessionhandlev2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.GetSessionHandleV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GetSessionHandleV2'.")

check_valid('tf.raw_ops.GetSessionHandleV2', generated_inputs['tf.raw_ops.GetSessionHandleV2'], lib="tf", suffix=0)
