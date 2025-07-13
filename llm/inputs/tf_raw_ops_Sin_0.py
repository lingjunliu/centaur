
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sin_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar
    x = np.array(0.0, dtype=np.float32)
    input_dict = {"x": x, "name": "sin_0"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 1D array
    x = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], dtype=np.float32)
    input_dict = {"x": x, "name": "sin_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 2D array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "sin_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, scalar
    x = np.array(1.0, dtype=np.float64)
    input_dict = {"x": x, "name": "sin_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64, 1D array
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    input_dict = {"x": x, "name": "sin_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float16, scalar
    x = np.array(0.5, dtype=np.float16)
    input_dict = {"x": x, "name": "sin_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float16, 2D array
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float16)
    input_dict = {"x": x, "name": "sin_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16, scalar
    x = np.array(0.7, dtype=np.float16)
    input_dict = {"x": x, "name": "sin_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float16, 1D array
    x = np.array([-0.5, 0.0, 0.5], dtype=np.float16)
    input_dict = {"x": x, "name": "sin_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex64, scalar
    x = np.array(1+1j, dtype=np.complex64)
    input_dict = {"x": x, "name": "sin_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: complex128, scalar
    x = np.array(1+1j, dtype=np.complex128)
    input_dict = {"x": x, "name": "sin_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: float32, 3D array
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"x": x, "name": "sin_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Sin"] = tf_raw_ops_sin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Sin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Sin'.")

check_valid('tf.raw_ops.Sin', generated_inputs['tf.raw_ops.Sin'], lib="tf", suffix=0)
