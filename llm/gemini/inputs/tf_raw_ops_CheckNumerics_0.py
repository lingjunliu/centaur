
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_check_numerics_inputs():
    list_of_inputs = []

    # Input 1: Valid float32 tensor with no NaN or Inf
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    message = "Test 1"
    input_dict = {"tensor": tensor, "message": message, "name": "check_numerics_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Valid float64 tensor with no NaN or Inf
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    message = "Test 2"
    input_dict = {"tensor": tensor, "message": message, "name": "check_numerics_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bfloat16 tensor with no NaN or Inf
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    message = "Test 3"
    input_dict = {"tensor": tensor, "message": message, "name": "check_numerics_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half tensor with no NaN or Inf
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    message = "Test 4"
    input_dict = {"tensor": tensor, "message": message, "name": "check_numerics_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: float32 multi-dimensional array
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    message = "Test 5"
    input_dict = {"tensor": tensor, "message": message, "name": "check_numerics_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: float64 single element
    tensor = np.array(5.0, dtype=np.float64)
    message = "Test 6"
    input_dict = {"tensor": tensor, "message": message, "name": "check_numerics_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: float32 with a large number
    tensor = np.array([1e9], dtype=np.float32)
    message = "Test 7"
    input_dict = {"tensor": tensor, "message": message, "name": "check_numerics_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bfloat16 tensor with small values
    tensor = np.array([1e-5, 2e-5, 3e-5], dtype=np.float16)
    message = "Test 8"
    input_dict = {"tensor": tensor, "message": message, "name": "check_numerics_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: half 2D array
    tensor = np.array([[1e-5, 2e-5], [3e-5, 4e-5]], dtype=np.float16)
    message = "Test 9"
    input_dict = {"tensor": tensor, "message": message, "name": "check_numerics_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: float64 with a negative number
    tensor = np.array([-5.0], dtype=np.float64)
    message = "Test 10"
    input_dict = {"tensor": tensor, "message": message, "name": "check_numerics_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.CheckNumerics"] = tf_raw_ops_check_numerics_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.CheckNumerics' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.CheckNumerics'.")

check_valid('tf.raw_ops.CheckNumerics', generated_inputs['tf.raw_ops.CheckNumerics'], lib="tf", suffix=0)
