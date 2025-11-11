
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_bitwise_xor_inputs():
    list_of_inputs = []

    
    # Input 1, valid
    x = np.array([0, 5, 3, 14], dtype=np.int8)
    y = np.array([5, 0, 7, 11], dtype=np.int8)
    name = "test1"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([5, 0, 7, 11], dtype=np.uint8)
    y = np.array([0, 5, 3, 14], dtype=np.uint8)
    name = "test2"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([1, 2, 3], dtype=np.int16)
    y = np.array([3, 2, 1], dtype=np.int16)
    name = "test3"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([0, 5, 3], dtype=np.uint16)
    y = np.array([5, 0, 7], dtype=np.uint16)
    name = "test4"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([4, 3, 2, 1], dtype=np.int32)
    name = "test5"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([0, 1, 2], dtype=np.uint32)
    y = np.array([2, 1, 0], dtype=np.uint32)
    name = "test6"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([3, 2, 1], dtype=np.int64)
    name = "test7"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([0, 1, 2], dtype=np.uint64)
    y = np.array([2, 1, 0], dtype=np.uint64)
    name = "test8"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - negative values
    x = np.array([-1, 2, -3], dtype=np.int8)
    y = np.array([1, -2, 3], dtype=np.int8)
    name = "test9"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - multi-dimensional arrays
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    y = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.int16)
    name = "test10"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.bitwise.bitwise_xor"] = tf_bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.bitwise.bitwise_xor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.bitwise.bitwise_xor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.bitwise.bitwise_xor', generated_inputs['tf.bitwise.bitwise_xor'], lib="tf", suffix=0)
