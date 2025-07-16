
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Erfinv_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([0.0, 0.5, 0.9], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    x = np.array([[-0.2, 0.3], [0.7, -0.9]], dtype=np.float64)
    input_dict = {"x": x, "name": "my_erfinv"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16, scalar
    x = np.array(0.8, dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float16, 3D array
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float16)
    input_dict = {"x": x, "name": "another_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, large values (close to 1 and -1)
    x = np.array([0.99, -0.95, 0.999], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: float64, zero value
    x = np.array(0.0, dtype=np.float64)
    input_dict = {"x": x, "name": "zero_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float16, all positive
    x = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: float16, negative and positive
    x = np.array([-0.3, 0.5, -0.7, 0.9], dtype=np.float16)
    input_dict = {"x": x, "name": "mixed_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, different shape
    x = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, scalar value with name
    x = np.array(-0.5, dtype=np.float64)
    input_dict = {"x": x, "name": "scalar_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Erfinv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Erfinv'.")

check_valid('tf.raw_ops.Erfinv', generated_inputs['tf.raw_ops.Erfinv'], lib="tf", suffix=0)
