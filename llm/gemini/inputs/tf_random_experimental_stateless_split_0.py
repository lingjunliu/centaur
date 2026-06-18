
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_experimental_stateless_split_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'seed': np.array([1, 2], dtype=np.int32),
        'num': 2,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'seed': np.array([42, 100], dtype=np.int32),
        'num': 3,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'seed': np.array([-1, -2], dtype=np.int32),
        'num': 5,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'seed': np.array([0, 0], dtype=np.int32),
        'num': 10,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'seed': np.array([2147483647, -2147483648], dtype=np.int32),
        'num': 1,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'seed': np.array([10, 20], dtype=np.int64),
        'num': 4,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'seed': np.array([-999, 999], dtype=np.int64),
        'num': 2,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'seed': np.array([123456789, 987654321], dtype=np.int64),
        'num': 8,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'seed': np.array([1, 1], dtype=np.int32),
        'num': 6,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'seed': np.array([987654, 3210], dtype=np.int32),
        'num': 12,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.experimental.stateless_split"] = tf_random_experimental_stateless_split_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.experimental.stateless_split' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.experimental.stateless_split'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.experimental.stateless_split', generated_inputs['tf.random.experimental.stateless_split'], lib="tf", suffix=0)
