
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_atanh_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar
    x = np.float32(0.5)
    input_dict = {"x": x, "name": "atanh_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 1D array
    x = np.array([-0.9, -0.5, 0, 0.5, 0.9], dtype=np.float32)
    input_dict = {"x": x, "name": "atanh_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 2D array
    x = np.array([[-0.8, -0.2], [0.2, 0.8]], dtype=np.float32)
    input_dict = {"x": x, "name": "atanh_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, scalar
    x = np.float64(0.7)
    input_dict = {"x": x, "name": "atanh_scalar_double"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64, 1D array
    x = np.array([-0.95, -0.25, 0.25, 0.95], dtype=np.float64)
    input_dict = {"x": x, "name": "atanh_1d_double"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, 2D array
    x = np.array([[-0.75, -0.35], [0.35, 0.75]], dtype=np.float64)
    input_dict = {"x": x, "name": "atanh_2d_double"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: complex64
    x = np.array([0.1 + 0.1j, 0.5 - 0.2j, -0.3 + 0.4j], dtype=np.complex64)
    input_dict = {"x": x, "name": "atanh_complex64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex128
    x = np.array([-0.2 + 0.3j, 0.4 - 0.1j], dtype=np.complex128)
    input_dict = {"x": x, "name": "atanh_complex128"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half
    x = np.array([-0.1, 0.3, 0.7], dtype=np.float16)
    input_dict = {"x": x, "name": "atanh_half"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, 3D array
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[-0.1, -0.2], [-0.3, -0.4]]], dtype=np.float32)
    input_dict = {"x": x, "name": "atanh_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Atanh"] = tf_raw_ops_atanh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Atanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Atanh'.")

check_valid('tf.raw_ops.Atanh', generated_inputs['tf.raw_ops.Atanh'], lib="tf", suffix=0)
