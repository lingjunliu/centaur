
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_normal_initializer_inputs():
    """
    This function generates a list of valid inputs for the tf.random_normal_initializer API.
    The API returns a callable initializer. The testing framework requires specifying
    the arguments for this callable in a special '__call_args__' dictionary.
    """
    list_of_inputs = []

    # Input 1: Basic case
    input_dict = {
        'mean': 0.0,
        'stddev': 1.0,
        'seed': 42,
        '__call_args__': {
            'shape': [2, 2],
            'dtype': np.float32
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative mean, different dtype
    input_dict = {
        'mean': -10.0,
        'stddev': 2.0,
        'seed': 1,
        '__call_args__': {
            'shape': [5],
            'dtype': np.float64
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero standard deviation
    input_dict = {
        'mean': 5.0,
        'stddev': 0.0,
        'seed': 123,
        '__call_args__': {
            'shape': [3, 4],
            'dtype': np.float32
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large standard deviation
    input_dict = {
        'mean': 0.0,
        'stddev': 100.0,
        'seed': 7,
        '__call_args__': {
            'shape': [1, 1, 1],
            'dtype': np.float32
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar shape
    input_dict = {
        'mean': 1.2,
        'stddev': 3.4,
        'seed': 99,
        '__call_args__': {
            'shape': [],
            'dtype': np.float32
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Zero seed
    input_dict = {
        'mean': 0.0,
        'stddev': 1.0,
        'seed': 0,
        '__call_args__': {
            'shape': [8],
            'dtype': np.float32
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: High precision float values
    input_dict = {
        'mean': 3.14159,
        'stddev': 2.71828,
        'seed': 314,
        '__call_args__': {
            'shape': [2, 5],
            'dtype': np.float32
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty tensor shape
    input_dict = {
        'mean': 1.0,
        'stddev': 1.0,
        'seed': 55,
        '__call_args__': {
            'shape': [10, 0],
            'dtype': np.float32
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float16 dtype
    input_dict = {
        'mean': 0.0,
        'stddev': 1.0,
        'seed': 101,
        '__call_args__': {
            'shape': [4, 4],
            'dtype': np.float16
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: large dimensions
    input_dict = {
        'mean': -0.5,
        'stddev': 1.5,
        'seed': 2023,
        '__call_args__': {
            'shape': [1, 32, 32, 3],
            'dtype': np.float32
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random_normal_initializer"] = tf_random_normal_initializer_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random_normal_initializer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random_normal_initializer'.")

check_valid('tf.random_normal_initializer', generated_inputs['tf.random_normal_initializer'], lib="tf", suffix=0)
