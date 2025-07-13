
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_log1p_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar
    x = np.array(0.5, dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 1D array
    x = np.array([-0.5, 0, 1.0, 2.5], dtype=np.float32)
    input_dict = {"x": x, "name": "log1p_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, 2D array
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    input_dict = {"x": x, "name": "log1p_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float16, 3D array
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float16)
    input_dict = {"x": x, "name": "log1p_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: half, large value
    x = np.array(10000.0, dtype=np.float16)
    input_dict = {"x": x, "name": "log1p_large"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex64, 1D array
    x = np.array([1+1j, 2-2j, 3+0j], dtype=np.complex64)
    input_dict = {"x": x, "name": "log1p_complex64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex128, 2D array
    x = np.array([[1+0j, 0-1j], [-1+1j, -1-1j]], dtype=np.complex128)
    input_dict = {"x": x, "name": "log1p_complex128"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, name is empty string
    x = np.array([0.5, 1.5], dtype=np.float32)
    input_dict = {"x": x, "name": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, all zeros
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"x": x, "name": "log1p_zeros"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: float32, all negative values (close to -1)
    x = np.array([-0.99, -0.9, -0.5], dtype=np.float32)
    input_dict = {"x": x, "name": "log1p_negatives"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Log1p"] = tf_raw_ops_log1p_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Log1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Log1p'.")

check_valid('tf.raw_ops.Log1p', generated_inputs['tf.raw_ops.Log1p'], lib="tf", suffix=0)
