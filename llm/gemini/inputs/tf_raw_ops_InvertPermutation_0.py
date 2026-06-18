
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_InvertPermutation_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        'name': 'invert_1',
        'x': np.array([0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {
        'name': 'invert_2',
        'x': np.array([1, 0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'name': 'invert_3',
        'x': np.array([2, 0, 1], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'name': 'invert_4',
        'x': np.array([3, 4, 0, 2, 1], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'name': 'invert_5',
        'x': np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'name': 'invert_6',
        'x': np.array([5, 4, 3, 2, 1, 0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'name': 'invert_7',
        'x': np.array([1, 3, 0, 2], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'name': 'invert_8',
        'x': np.array([2, 1, 0, 4, 3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'name': 'invert_9',
        'x': np.array([4, 3, 2, 1, 0, 5, 6], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'name': 'invert_10',
        'x': np.array([0, 2, 1, 3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.InvertPermutation"] = tf_raw_ops_InvertPermutation_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.InvertPermutation' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.InvertPermutation'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.InvertPermutation', generated_inputs['tf.raw_ops.InvertPermutation'], lib="tf", suffix=0)
