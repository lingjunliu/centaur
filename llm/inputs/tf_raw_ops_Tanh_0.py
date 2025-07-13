
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_tanh_inputs():
    list_of_inputs = []

    # Input 2: half, 2D array
    x = np.array([[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]], dtype=np.float16)
    input_dict = {"x": x, "name": "tanh_half"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 3D array
    x = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"x": x, "name": "tanh_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, scalar
    x = np.array(-1.5, dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, 1D array
    x = np.array([1 + 1j, 2 - 2j, -1 + 0j], dtype=np.complex64)
    input_dict = {"x": x, "name": "tanh_complex64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, 2D array
    x = np.array([[1 + 0j, 0 - 1j], [-1 - 1j, 2 + 2j]], dtype=np.complex128)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, 4D array, with large values
    x = np.random.randn(2, 2, 2, 2).astype(np.float32) * 100
    input_dict = {"x": x, "name": "tanh_large"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, 1D array with zeros
    x = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half, 1D array with inf and -inf
    x = np.array([np.inf, -np.inf, 0.0], dtype=np.float16)
    input_dict = {"x": x, "name": "tanh_inf"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128, scalar 0j
    x = np.array(0j, dtype=np.complex128)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: float32, empty array
    x = np.array([], dtype=np.float32)
    input_dict = {"x": x, "name": "tanh_empty"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Tanh"] = tf_raw_ops_tanh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Tanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Tanh'.")

check_valid('tf.raw_ops.Tanh', generated_inputs['tf.raw_ops.Tanh'], lib="tf", suffix=0)
