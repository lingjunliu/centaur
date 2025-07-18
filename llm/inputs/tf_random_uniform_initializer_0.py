
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def generate_tf_random_uniform_initializer_inputs():
    """
    Generates a list of valid inputs for the tf.random_uniform_initializer function.
    This API returns a callable initializer. The test harness expects arguments
    for the initializer's constructor at the top level, and arguments for the
    subsequent call to be nested under the 'call_args' key. The dtype is
    specified as a TensorFlow DType.
    """
    list_of_inputs = []

    # Input 1: Standard small range, float32
    list_of_inputs.append(copy.deepcopy({
        'minval': -0.05,
        'maxval': 0.05,
        'seed': 1,
        'call_args': {'shape': [3, 3], 'dtype': tf.float32}
    }))

    # Input 2: Range [0, 1], float64
    list_of_inputs.append(copy.deepcopy({
        'minval': 0.0,
        'maxval': 1.0,
        'seed': 42,
        'call_args': {'shape': [10], 'dtype': tf.float64}
    }))

    # Input 3: Negative range, 3D shape, float32
    list_of_inputs.append(copy.deepcopy({
        'minval': -1.0,
        'maxval': -0.1,
        'seed': 123,
        'call_args': {'shape': [2, 3, 4], 'dtype': tf.float32}
    }))

    # Input 4: Wider range, float16
    list_of_inputs.append(copy.deepcopy({
        'minval': -10.0,
        'maxval': 10.0,
        'seed': 7,
        'call_args': {'shape': [5, 5], 'dtype': tf.float16}
    }))

    # Input 5: Very small positive range
    list_of_inputs.append(copy.deepcopy({
        'minval': 1e-6,
        'maxval': 1e-5,
        'seed': 100,
        'call_args': {'shape': [1, 10], 'dtype': tf.float32}
    }))

    # Input 6: Large positive range
    list_of_inputs.append(copy.deepcopy({
        'minval': 100.0,
        'maxval': 200.0,
        'seed': 200,
        'call_args': {'shape': [8], 'dtype': tf.float64}
    }))

    # Input 7: Scalar output
    list_of_inputs.append(copy.deepcopy({
        'minval': 0.9,
        'maxval': 1.0,
        'seed': 300,
        'call_args': {'shape': [], 'dtype': tf.float32}
    }))

    # Input 8: Large negative range, large seed
    list_of_inputs.append(copy.deepcopy({
        'minval': -1000.0,
        'maxval': -500.0,
        'seed': 9999,
        'call_args': {'shape': [2, 2], 'dtype': tf.float16}
    }))

    # Input 9: Seed set to 0, 4D shape
    list_of_inputs.append(copy.deepcopy({
        'minval': -2.0,
        'maxval': 2.0,
        'seed': 0,
        'call_args': {'shape': [1, 2, 3, 1], 'dtype': tf.float32}
    }))

    # Input 10: minval and maxval are very close
    list_of_inputs.append(copy.deepcopy({
        'minval': 0.49999,
        'maxval': 0.5,
        'seed': 500,
        'call_args': {'shape': [4, 1], 'dtype': tf.float64}
    }))

    return list_of_inputs

generated_inputs["tf.random_uniform_initializer"] = generate_tf_random_uniform_initializer_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random_uniform_initializer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random_uniform_initializer'.")

check_valid('tf.random_uniform_initializer', generated_inputs['tf.random_uniform_initializer'], lib="tf", suffix=0)
