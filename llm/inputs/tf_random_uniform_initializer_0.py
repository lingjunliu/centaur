
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_tf_random_uniform_initializer_inputs():
    list_of_inputs = []

    # Input 1: Basic case, 1D float32 tensor
    input_dict_1 = {
        'minval': -0.05,
        'maxval': 0.05,
        'seed': 0,
        '__call__': {
            'args': ([10],),
            'kwargs': {'dtype': np.float32}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Positive range with a seed, 2D shape, float64
    input_dict_2 = {
        'minval': 0.1,
        'maxval': 0.2,
        'seed': 1,
        '__call__': {
            'args': ([5, 5],),
            'kwargs': {'dtype': np.float64}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Negative range with a seed, 3D shape
    input_dict_3 = {
        'minval': -10.0,
        'maxval': -5.0,
        'seed': 42,
        '__call__': {
            'args': ([2, 3, 4],),
            'kwargs': {'dtype': np.float32}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Standard symmetric range with a seed, 1D shape of size 1
    input_dict_4 = {
        'minval': -1.0,
        'maxval': 1.0,
        'seed': 123,
        '__call__': {
            'args': ([1],),
            'kwargs': {'dtype': np.float32}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Wide range, float64
    input_dict_5 = {
        'minval': -1000.0,
        'maxval': 1000.0,
        'seed': 999,
        '__call__': {
            'args': ([8, 2],),
            'kwargs': {'dtype': np.float64}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Narrow range with a seed, large 1D shape
    input_dict_6 = {
        'minval': 0.499,
        'maxval': 0.5,
        'seed': 2023,
        '__call__': {
            'args': ([100],),
            'kwargs': {'dtype': np.float32}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Zero as lower bound
    input_dict_7 = {
        'minval': 0.0,
        'maxval': 1.0,
        'seed': 7,
        '__call__': {
            'args': ([4, 4],),
            'kwargs': {'dtype': np.float32}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Zero as upper bound, float64, 3D shape
    input_dict_8 = {
        'minval': -1.0,
        'maxval': 0.0,
        'seed': 8,
        '__call__': {
            'args': ([3, 1, 5],),
            'kwargs': {'dtype': np.float64}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Large magnitude values
    input_dict_9 = {
        'minval': 1000000.0,
        'maxval': 2000000.0,
        'seed': 99,
        '__call__': {
            'args': ([16],),
            'kwargs': {'dtype': np.float32}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Scalar (0D) tensor
    input_dict_10 = {
        'minval': -1.0,
        'maxval': 1.0,
        'seed': 101,
        '__call__': {
            'args': ([],),
            'kwargs': {'dtype': np.float32}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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
