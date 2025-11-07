
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_floormod_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([2, 2, 2, 2], dtype=np.int32)
    name = "test1"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([10, -5, 8], dtype=np.float32)
    y = np.array([3.0, -2.0, 4.0], dtype=np.float32)
    name = "test2"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([[2, 3], [4, 5]], dtype=np.int64)
    name = "test3"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([0.5, -1.5, 2.5], dtype=np.float64)
    y = np.array([1.0, -1.0, 3.0], dtype=np.float64)
    name = "test4"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([100, -200, 300], dtype=np.int8)
    y = np.array([7, -8, 9], dtype=np.int8)
    name = "test5"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    y = np.array([3, 3, 3, 3, 3], dtype=np.int16)
    name = "test6"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([[-10, 20], [-30, 40]], dtype=np.float32)
    y = np.array([5.0, -10.0], dtype=np.float32)
    name = "test7"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([3, 3, 3], dtype=np.uint8)
    name = "test8"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.int32)
    name = "test9"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([1, 2, 3], dtype=np.float32)
    y = np.array([3, 3, 3], dtype=np.float32)
    name = "test10"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.floormod"] = tf_math_floormod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.floormod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.floormod'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.floormod', generated_inputs['tf.math.floormod'], lib="tf", suffix=0)
