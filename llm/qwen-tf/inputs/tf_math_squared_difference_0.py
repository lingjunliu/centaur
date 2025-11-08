
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_squared_difference_inputs():
    list_of_inputs = []
    
    # Input 1: Valid - float32 tensors
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Valid - int32 tensors
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[2, 3], [4, 5]], dtype=np.int32)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Valid - float64 tensors
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Valid - complex64 tensors
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    y = np.array([[2+4j, 6+8j], [10+12j, 14+16j]], dtype=np.complex64)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Valid - complex128 tensors
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    y = np.array([[2+4j, 6+8j], [10+12j, 14+16j]], dtype=np.complex128)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Valid - mixed negative values
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([-2, -4], dtype=np.int32)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Valid - broadcasting tensors
    x = np.array([1, 2], dtype=np.float32)
    y = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Valid - scalar tensors
    x = np.array(5, dtype=np.int32)
    y = np.array(3, dtype=np.int32)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Valid - float64 with negative values
    x = np.array([-1.0, -2.0], dtype=np.float64)
    y = np.array([1.0, 2.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Valid - mixed dimensions
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    y = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.squared_difference"] = tf_math_squared_difference_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.squared_difference' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.squared_difference'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.squared_difference', generated_inputs['tf.math.squared_difference'], lib="tf", suffix=0)
