
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_atan_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D
    x = np.array([0.0, 1.0, -1.0, np.sqrt(3), -np.sqrt(3)], dtype=np.float32)
    input_dict = {"x": x, "name": "atan_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D
    x = np.array([[0.5, -0.5], [1.5, -1.5]], dtype=np.float64)
    input_dict = {"x": x, "name": "atan_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bfloat16, 3D
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float16)
    input_dict = {"x": x, "name": "atan_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half, 1D with larger values
    x = np.array([10.0, -10.0, 100.0, -100.0], dtype=np.float16)
    input_dict = {"x": x, "name": "atan_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, 2D
    x = np.array([[1+1j, 2-2j], [3+0j, 0-4j]], dtype=np.complex64)
    input_dict = {"x": x, "name": "atan_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, 1D
    x = np.array([1j, -1j, 2+3j, -2-3j], dtype=np.complex128)
    input_dict = {"x": x, "name": "atan_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, scalar
    x = np.array(0.707, dtype=np.float32)
    input_dict = {"x": x, "name": "atan_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, 3D with different values
    x = np.array([[[0.9, -0.1], [0.2, -0.8]], [[0.6, -0.4], [0.7, -0.3]]], dtype=np.float64)
    input_dict = {"x": x, "name": "atan_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, empty array
    x = np.array([], dtype=np.float32)
    input_dict = {"x": x, "name": "atan_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, 4D array
    x = np.random.rand(2, 2, 2, 2).astype(np.float32)
    input_dict = {"x": x, "name": "atan_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Atan"] = tf_raw_ops_atan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Atan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Atan'.")

check_valid('tf.raw_ops.Atan', generated_inputs['tf.raw_ops.Atan'], lib="tf", suffix=0)
