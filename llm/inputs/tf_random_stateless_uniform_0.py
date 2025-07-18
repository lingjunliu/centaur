
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def get_stateless_uniform_inputs():
    """
    Generates a list of valid inputs for tf.random.stateless_uniform.
    """
    list_of_inputs = []

    # Case 1: Basic float32, default range [0, 1)
    input_dict_1 = {
        'shape': np.array([10], dtype=np.int32),
        'seed': np.array([1, 2], dtype=np.int32),
        'minval': np.array(0.0, dtype=np.float32),
        'maxval': np.array(1.0, dtype=np.float32),
        'dtype': np.float32,
        'name': 'basic_float32',
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: 2D float32 with a custom range
    input_dict_2 = {
        'shape': np.array([3, 4], dtype=np.int32),
        'seed': np.array([3, 4], dtype=np.int32),
        'minval': np.array(-5.0, dtype=np.float32),
        'maxval': np.array(5.0, dtype=np.float32),
        'dtype': np.float32,
        'name': '2d_float32_custom_range',
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Basic float64
    input_dict_3 = {
        'shape': np.array([5], dtype=np.int32),
        'seed': np.array([5, 6], dtype=np.int32),
        'minval': np.array(0.0, dtype=np.float64),
        'maxval': np.array(1.0, dtype=np.float64),
        'dtype': np.float64,
        'name': 'basic_float64',
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: 2D float64 with a custom range
    input_dict_4 = {
        'shape': np.array([2, 2], dtype=np.int32),
        'seed': np.array([7, 8], dtype=np.int32),
        'minval': np.array(100.0, dtype=np.float64),
        'maxval': np.array(200.0, dtype=np.float64),
        'dtype': np.float64,
        'name': '2d_float64_custom_range',
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Basic int32, explicitly setting alg to 'philox' to avoid CPU counter error.
    input_dict_5 = {
        'shape': np.array([8], dtype=np.int32),
        'seed': np.array([9, 10], dtype=np.int32),
        'minval': np.array(0, dtype=np.int32),
        'maxval': np.array(100, dtype=np.int32),
        'dtype': np.int32,
        'name': 'basic_int32_philox',
        'alg': 'philox'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: 2D int32 with a negative range, explicitly setting alg to 'threefry'.
    input_dict_6 = {
        'shape': np.array([4, 3], dtype=np.int32),
        'seed': np.array([11, 12], dtype=np.int32),
        'minval': np.array(-50, dtype=np.int32),
        'maxval': np.array(-10, dtype=np.int32),
        'dtype': np.int32,
        'name': '2d_int32_negative_range_threefry',
        'alg': 'threefry'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Basic int64 with int64 seed, explicitly setting alg to 'philox'.
    input_dict_7 = {
        'shape': np.array([6], dtype=np.int32),
        'seed': np.array([13, 14], dtype=np.int64),
        'minval': np.array(1000, dtype=np.int64),
        'maxval': np.array(2000, dtype=np.int64),
        'dtype': np.int64,
        'name': 'basic_int64_philox',
        'alg': 'philox'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: 2D int64 with a large range, explicitly setting alg to 'threefry'.
    input_dict_8 = {
        'shape': np.array([2, 5], dtype=np.int32),
        'seed': np.array([15, 16], dtype=np.int64),
        'minval': np.array(-100000, dtype=np.int64),
        'maxval': np.array(100000, dtype=np.int64),
        'dtype': np.int64,
        'name': '2d_int64_large_range_threefry',
        'alg': 'threefry'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: Using "philox" algorithm with floats
    input_dict_9 = {
        'shape': np.array([3, 3], dtype=np.int32),
        'seed': np.array([17, 18], dtype=np.int32),
        'minval': np.array(0.0, dtype=np.float32),
        'maxval': np.array(1.0, dtype=np.float32),
        'dtype': np.float32,
        'name': 'philox_alg_float',
        'alg': 'philox'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: Using "threefry" algorithm with integers
    input_dict_10 = {
        'shape': np.array([12], dtype=np.int32),
        'seed': np.array([19, 20], dtype=np.int32),
        'minval': np.array(0, dtype=np.int32),
        'maxval': np.array(10, dtype=np.int32),
        'dtype': np.int32,
        'name': 'threefry_alg_int',
        'alg': 'threefry'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Case 11: Using float16 dtype
    input_dict_11 = {
        'shape': np.array([2, 4], dtype=np.int32),
        'seed': np.array([21, 22], dtype=np.int32),
        'minval': np.array(-1.0, dtype=np.float16),
        'maxval': np.array(1.0, dtype=np.float16),
        'dtype': np.float16,
        'name': 'float16_type',
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Case 12: Scalar float output
    input_dict_12 = {
        'shape': np.array([], dtype=np.int32),
        'seed': np.array([23, 24], dtype=np.int32),
        'minval': np.array(5.0, dtype=np.float32),
        'maxval': np.array(6.0, dtype=np.float32),
        'dtype': np.float32,
        'name': 'scalar_float_output',
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.random.stateless_uniform"] = get_stateless_uniform_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_uniform' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_uniform'.")

check_valid('tf.random.stateless_uniform', generated_inputs['tf.random.stateless_uniform'], lib="tf", suffix=0)
