
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_uniform_inputs():
    list_of_inputs = []

    # Input 1: Basic float32, alg='threefry'
    input_dict = {
        'shape': np.array([2, 2], dtype=np.int32),
        'seed': np.array([1, 2], dtype=np.int64),
        'minval': np.array(0.0, dtype=np.float32),
        'maxval': np.array(1.0, dtype=np.float32),
        'dtype': np.float32,
        'name': 'float32_threefry',
        'alg': 'threefry'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic float64, alg='threefry'
    input_dict = {
        'shape': np.array([3], dtype=np.int32),
        'seed': np.array([3, 4], dtype=np.int64),
        'minval': np.array(-10.0, dtype=np.float64),
        'maxval': np.array(10.0, dtype=np.float64),
        'dtype': np.float64,
        'name': 'float64_threefry',
        'alg': 'threefry'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float16 dtype, alg='threefry'
    input_dict = {
        'shape': np.array([2, 5], dtype=np.int32),
        'seed': np.array([5, 6], dtype=np.int64),
        'minval': np.array(-1.0, dtype=np.float16),
        'maxval': np.array(1.0, dtype=np.float16),
        'dtype': np.float16,
        'name': 'float16_threefry',
        'alg': 'threefry'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: bfloat16 dtype, alg='threefry'
    bfloat16_dtype = tf.bfloat16.as_numpy_dtype
    input_dict = {
        'shape': np.array([3, 2], dtype=np.int32),
        'seed': np.array([7, 8], dtype=np.int64),
        'minval': np.array(-100.0, dtype=bfloat16_dtype),
        'maxval': np.array(100.0, dtype=bfloat16_dtype),
        'dtype': bfloat16_dtype,
        'name': 'bfloat16_threefry',
        'alg': 'threefry'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: broadcasting minval/maxval for floats, alg='threefry'
    input_dict = {
        'shape': np.array([2, 2], dtype=np.int32),
        'seed': np.array([9, 10], dtype=np.int64),
        'minval': np.array([-10., 0.], dtype=np.float32),
        'maxval': np.array([0., 10.], dtype=np.float32),
        'dtype': np.float32,
        'name': 'broadcast_minmax_threefry',
        'alg': 'threefry'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Scalar output (empty shape), alg='threefry'
    input_dict = {
        'shape': np.array([], dtype=np.int32),
        'seed': np.array([11, 12], dtype=np.int64),
        'minval': np.array(10.0, dtype=np.float32),
        'maxval': np.array(20.0, dtype=np.float32),
        'dtype': np.float32,
        'name': 'scalar_output_threefry',
        'alg': 'threefry'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty tensor output (one dimension is 0), alg='threefry'
    input_dict = {
        'shape': np.array([5, 0], dtype=np.int32),
        'seed': np.array([13, 14], dtype=np.int64),
        'minval': np.array(0.0, dtype=np.float32),
        'maxval': np.array(1.0, dtype=np.float32),
        'dtype': np.float32,
        'name': 'empty_tensor_output_threefry',
        'alg': 'threefry'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int32 output with int32 seed, alg='threefry'
    input_dict = {
        'shape': np.array([10], dtype=np.int32),
        'seed': np.array([15, 16], dtype=np.int32),
        'minval': np.array(0, dtype=np.int32),
        'maxval': np.array(100, dtype=np.int32),
        'dtype': np.int32,
        'name': 'int32_int32seed_threefry',
        'alg': 'threefry'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int64 output with int32 seed, alg='threefry'
    input_dict = {
        'shape': np.array([8], dtype=np.int32),
        'seed': np.array([17, 18], dtype=np.int32),
        'minval': np.array(-5000, dtype=np.int64),
        'maxval': np.array(5000, dtype=np.int64),
        'dtype': np.int64,
        'name': 'int64_int32seed_threefry',
        'alg': 'threefry'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int32 output with int64 seed, alg='threefry'
    input_dict = {
        'shape': np.array([4, 4], dtype=np.int32),
        'seed': np.array([19, 20], dtype=np.int64),
        'minval': np.array(-10, dtype=np.int32),
        'maxval': np.array(10, dtype=np.int32),
        'dtype': np.int32,
        'name': 'int32_int64seed_threefry',
        'alg': 'threefry'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: int64 output with int64 seed, alg='threefry'
    input_dict = {
        'shape': np.array([5], dtype=np.int32),
        'seed': np.array([21, 22], dtype=np.int64),
        'minval': np.array(1000000, dtype=np.int64),
        'maxval': np.array(2000000, dtype=np.int64),
        'dtype': np.int64,
        'name': 'int64_int64seed_threefry',
        'alg': 'threefry'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_uniform"] = tf_random_stateless_uniform_inputs()

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
