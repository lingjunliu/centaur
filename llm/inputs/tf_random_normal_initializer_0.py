
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_normal_initializer_inputs():
    list_of_inputs = []

    def create_input(mean, stddev, seed, shape, dtype):
        input_dict = {
            'mean': float(mean),
            'stddev': float(stddev),
            'seed': int(seed)
        }
        # The error indicates the test harness expects arguments for the callable
        # returned by the initializer. We hypothesize the key for these
        # arguments is 'call_args'.
        input_dict['call_args'] = {
            'shape': shape,
            'dtype': dtype,
        }
        return input_dict

    # Input 1: Basic 1D case, float32
    list_of_inputs.append(copy.deepcopy(create_input(0.0, 1.0, 1, [10], np.float32)))

    # Input 2: Negative mean, 2D shape, float32
    list_of_inputs.append(copy.deepcopy(create_input(-5.0, 2.0, 42, [5, 5], np.float32)))

    # Input 3: Zero stddev, 3D shape (results in a tensor of constants), float32
    list_of_inputs.append(copy.deepcopy(create_input(10.0, 0.0, 7, [2, 3, 4], np.float32)))

    # Input 4: Small stddev, float64 dtype
    list_of_inputs.append(copy.deepcopy(create_input(0.0, 1e-5, 100, [100], np.float64)))

    # Input 5: Large values, 1-element shape, float32
    list_of_inputs.append(copy.deepcopy(create_input(1000.0, 500.0, 2023, [1], np.float32)))

    # Input 6: Values from docs, 2D shape, float32
    list_of_inputs.append(copy.deepcopy(create_input(0.0, 0.05, 0, [8, 2], np.float32)))

    # Input 7: Negative seed, float64 dtype
    list_of_inputs.append(copy.deepcopy(create_input(-1.0, 1.5, -10, [4, 4], np.float64)))

    # Input 8: Fractional values, 3D shape, float32
    list_of_inputs.append(copy.deepcopy(create_input(3.14, 2.718, 99, [1, 1, 10], np.float32)))

    # Input 9: Scalar output (empty shape), float32
    list_of_inputs.append(copy.deepcopy(create_input(0.5, 0.5, 1337, [], np.float32)))

    # Input 10: Zero-sized dimension in shape, float32
    list_of_inputs.append(copy.deepcopy(create_input(0.0, 1.0, 2, [5, 0], np.float32)))

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
