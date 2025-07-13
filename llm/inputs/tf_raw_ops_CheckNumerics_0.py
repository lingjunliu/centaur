
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_check_numerics_inputs():
    list_of_inputs = []

    # Input 1
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    message = "Test 1"
    name = "CheckNumerics1"
    input_dict = {"tensor": tf.convert_to_tensor(tensor), "message": message, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    message = "Test 2"
    name = "CheckNumerics2"
    input_dict = {"tensor": tf.convert_to_tensor(tensor), "message": message, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    message = ""
    name = None
    input_dict = {"tensor": tf.convert_to_tensor(tensor), "message": message, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    message = "Test 4"
    name = ""
    input_dict = {"tensor": tf.convert_to_tensor(tensor), "message": message, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    message = "Test 5"
    name = "CheckNumerics5"
    input_dict = {"tensor": tf.convert_to_tensor(tensor), "message": message, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    message = "Test 6"
    name = "CheckNumerics6"
    input_dict = {"tensor": tf.convert_to_tensor(tensor), "message": message, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    message = "Test 7"
    name = "CheckNumerics7"
    input_dict = {"tensor": tf.convert_to_tensor(tensor), "message": message, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    tensor = np.array([1.0], dtype=np.float32)
    message = "Test 8"
    name = "CheckNumerics8"
    input_dict = {"tensor": tf.convert_to_tensor(tensor), "message": message, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    tensor = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    message = "Test 9"
    name = "CheckNumerics9"
    input_dict = {"tensor": tf.convert_to_tensor(tensor), "message": message, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    tensor = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64)
    message = "Test 10"
    name = "CheckNumerics10"
    input_dict = {"tensor": tf.convert_to_tensor(tensor), "message": message, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.CheckNumerics"] = tf_raw_ops_check_numerics_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.CheckNumerics' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.CheckNumerics'.")

check_valid('tf.raw_ops.CheckNumerics', generated_inputs['tf.raw_ops.CheckNumerics'], lib="tf", suffix=0)
