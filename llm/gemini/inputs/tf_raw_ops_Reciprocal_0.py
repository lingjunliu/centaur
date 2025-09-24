
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_reciprocal_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {"x": x, "name": "reciprocal_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    x = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "reciprocal_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: complex64, 1D array
    x = np.array([1 + 1j, 2 - 2j, 3 + 0j], dtype=np.complex64)
    input_dict = {"x": x, "name": "reciprocal_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex128, 2D array
    x = np.array([[1 + 0j, 0 - 1j], [-1 + 1j, 2 + 2j]], dtype=np.complex128)
    input_dict = {"x": x, "name": "reciprocal_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, 1D array, avoid 0
    x = np.array([1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    input_dict = {"x": x, "name": "reciprocal_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32, 3D array
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"x": x, "name": "reciprocal_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64, 2D array, avoid 0
    x = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "reciprocal_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16, 1D array
    x = np.array([1.0, 2.0, -3.0], dtype=np.float16)
    input_dict = {"x": x, "name": "reciprocal_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float16, 1D array
    x = np.array([1.0, -2.0, 3.0], dtype=np.float16)
    input_dict = {"x": x, "name": "reciprocal_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, 1D array
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {"x": x, "name": "reciprocal_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Reciprocal"] = tf_raw_ops_reciprocal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Reciprocal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Reciprocal'.")

check_valid('tf.raw_ops.Reciprocal', generated_inputs['tf.raw_ops.Reciprocal'], lib="tf", suffix=0)
