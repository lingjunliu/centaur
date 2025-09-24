
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def get_stateless_uniform_inputs():
    """
    Generates a list of valid inputs for tf.random.stateless_uniform.
    """
    list_of_inputs = []

    # Input 1: Basic float32, 2D output
    input_dict_1 = {
        'shape': np.array([3, 2], dtype=np.int32),
        'seed': np.array([1, 2], dtype=np.int32),
        'minval': np.array(0.0, dtype=np.float32),
        'maxval': np.array(1.0, dtype=np.float32),
        'dtype': np.float32,
        'name': 'float32_2d',
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic float64, 2D output
    input_dict_2 = {
        'shape': np.array([2, 2], dtype=np.int32),
        'seed': np.array([10, 20], dtype=np.int32),
        'minval': np.array(5.0, dtype=np.float64),
        'maxval': np.array(15.0, dtype=np.float64),
        'dtype': np.float64,
        'name': 'float64_2d',
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Negative float range, 3D output
    input_dict_3 = {
        'shape': np.array([2, 1, 3], dtype=np.int32),
        'seed': np.array([-1, -2], dtype=np.int32),
        'minval': np.array(-10.0, dtype=np.float32),
        'maxval': np.array(-5.0, dtype=np.float32),
        'dtype': np.float32,
        'name': 'float32_3d_negative',
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 'philox' algorithm, 2D output
    input_dict_4 = {
        'shape': np.array([3, 2], dtype=np.int32),
        'seed': np.array([7, 8], dtype=np.int32),
        'minval': np.array(0.0, dtype=np.float32),
        'maxval': np.array(1.0, dtype=np.float32),
        'dtype': np.float32,
        'name': 'philox_2d',
        'alg': 'philox'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Scalar output (shape length 0)
    input_dict_5 = {
        'shape': np.array([], dtype=np.int32),
        'seed': np.array([123, 456], dtype=np.int32),
        'minval': np.array(0.0, dtype=np.float32),
        'maxval': np.array(100.0, dtype=np.float32),
        'dtype': np.float32,
        'name': 'scalar_output',
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: float16 dtype, 2D output
    input_dict_6 = {
        'shape': np.array([2, 4], dtype=np.int32),
        'seed': np.array([5, 5], dtype=np.int32),
        'minval': np.array(0.0, dtype=np.float16),
        'maxval': np.array(1.0, dtype=np.float16),
        'dtype': np.float16,
        'name': 'float16_2d',
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: 'threefry' algorithm with float64, 2D output
    input_dict_7 = {
        'shape': np.array([3, 3], dtype=np.int32),
        'seed': np.array([2, 3], dtype=np.int32),
        'minval': np.array(0.0, dtype=np.float64),
        'maxval': np.array(2.0, dtype=np.float64),
        'dtype': np.float64,
        'name': 'threefry_float64_2d',
        'alg': 'threefry'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: broadcasting minval/maxval, 2D output
    input_dict_8 = {
        'shape': np.array([2, 3], dtype=np.int32),
        'seed': np.array([77, 88], dtype=np.int32),
        'minval': np.array([0., 10., 20.], dtype=np.float32),
        'maxval': np.array([[100.], [200.]], dtype=np.float32),
        'dtype': np.float32,
        'name': 'broadcastable_minmax_2d',
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: int32 with 2D shape
    input_dict_9 = {
        'shape': np.array([4, 4], dtype=np.int32),
        'seed': np.array([42, 1337], dtype=np.int32),
        'minval': np.array(10, dtype=np.int32),
        'maxval': np.array(100, dtype=np.int32),
        'dtype': np.int32,
        'name': 'int32_2d',
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: int64 with 3D shape
    input_dict_10 = {
        'shape': np.array([2, 2, 2], dtype=np.int32),
        'seed': np.array([99, 88], dtype=np.int32),
        'minval': np.array(-1000, dtype=np.int64),
        'maxval': np.array(1000, dtype=np.int64),
        'dtype': np.int64,
        'name': 'int64_3d',
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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
