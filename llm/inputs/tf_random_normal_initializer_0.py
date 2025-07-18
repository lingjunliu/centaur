
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_normal_initializer_inputs():
    """
    Generates a list of valid inputs for tf.random_normal_initializer.
    The test harness requires 'shape' and 'dtype' to be provided in the input
    dictionary to call the initializer object that the API returns. This is
    done via a nested dictionary with the key 'initializer_args'.
    """
    list_of_inputs = []

    def create_input_dict(mean, stddev, seed, shape, dtype):
        return {
            'mean': mean,
            'stddev': stddev,
            'seed': seed,
            'initializer_args': {
                'shape': shape,
                'dtype': dtype
            }
        }

    # Input 1: Standard case
    list_of_inputs.append(copy.deepcopy(create_input_dict(0.0, 1.0, 42, [10], np.float32)))

    # Input 2: Negative mean, 2D shape, float64
    list_of_inputs.append(copy.deepcopy(create_input_dict(-5.5, 2.0, 123, (3, 3), np.float64)))

    # Input 3: Zero stddev
    list_of_inputs.append(copy.deepcopy(create_input_dict(10.0, 0.0, 1, [2, 3, 4], np.float32)))

    # Input 4: Small stddev
    list_of_inputs.append(copy.deepcopy(create_input_dict(0.0, 0.001, 777, (5, 2), np.float32)))

    # Input 5: Scalar shape
    list_of_inputs.append(copy.deepcopy(create_input_dict(1000.0, 500.0, 2024, [], np.float64)))

    # Input 6: Zero seed
    list_of_inputs.append(copy.deepcopy(create_input_dict(0.0, 0.05, 0, [100], np.float32)))

    # Input 7: Large seed
    list_of_inputs.append(copy.deepcopy(create_input_dict(1.5, 0.25, 999999, (4, 4), np.float32)))

    # Input 8: Negative seed
    list_of_inputs.append(copy.deepcopy(create_input_dict(-2.0, 1.0, -10, (1, 1, 1), np.float64)))

    # Input 9: Fractional values
    list_of_inputs.append(copy.deepcopy(create_input_dict(3.14, 2.71, 314159, [8], np.float32)))

    # Input 10: Large negative mean
    list_of_inputs.append(copy.deepcopy(create_input_dict(-5000.0, 10.0, 1337, (2, 5), np.float32)))

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
