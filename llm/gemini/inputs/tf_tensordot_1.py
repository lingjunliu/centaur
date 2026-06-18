
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_tensordot_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        "a": np.random.randn(3, 4).astype(np.float32),
        "b": np.random.randn(4, 5).astype(np.float32),
        "axes": [[1], [0]],
        "name": "tensordot_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {
        "a": np.random.randn(3, 4, 5).astype(np.float32),
        "b": np.random.randn(4, 3, 2).astype(np.float32),
        "axes": [[1, 0], [0, 1]],
        "name": "tensordot_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {
        "a": np.random.randn(5, 5).astype(np.float64),
        "b": np.random.randn(5, 5).astype(np.float64),
        "axes": [[0], [0]],
        "name": "tensordot_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {
        "a": np.random.randn(2, 3, 4).astype(np.float32),
        "b": np.random.randn(3, 4, 5).astype(np.float32),
        "axes": [[1, 2], [0, 1]],
        "name": "tensordot_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        "a": np.random.randn(10).astype(np.float32),
        "b": np.random.randn(10).astype(np.float32),
        "axes": [[0], [0]],
        "name": "tensordot_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {
        "a": np.random.randn(2, 2, 2).astype(np.float64),
        "b": np.random.randn(2, 2, 2).astype(np.float64),
        "axes": [[0, 1, 2], [0, 1, 2]],
        "name": "tensordot_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {
        "a": np.random.randn(4, 2).astype(np.float32),
        "b": np.random.randn(2, 3).astype(np.float32),
        "axes": [[1], [0]],
        "name": "tensordot_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {
        "a": np.random.randn(1, 5, 1).astype(np.float64),
        "b": np.random.randn(5, 1, 1).astype(np.float64),
        "axes": [[1], [0]],
        "name": "tensordot_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {
        "a": np.random.randn(2, 3).astype(np.float32),
        "b": np.random.randn(3, 4).astype(np.float32),
        "axes": [[1], [0]],
        "name": "tensordot_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        "a": np.random.randn(3, 3, 3).astype(np.float32),
        "b": np.random.randn(3, 3, 3).astype(np.float32),
        "axes": [[2], [1]],
        "name": "tensordot_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.tensordot_1"] = tf_tensordot_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.tensordot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.tensordot_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.tensordot', generated_inputs['tf.tensordot_1'], lib="tf", suffix=1)
