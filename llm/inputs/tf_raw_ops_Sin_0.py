
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
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 1D array
    x = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, 2D array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32, 3D array
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: half, scalar
    x = np.array(1.0, dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex64, 1D array
    x = np.array([1.0 + 1j, 2.0 - 2j], dtype=np.complex64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex128, 2D array
    x = np.array([[1.0 + 1j, 2.0 - 2j], [3.0 + 3j, 4.0 - 4j]], dtype=np.complex128)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, negative values
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, large values
    x = np.array([1000.0, 2000.0, 3000.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: half, 2D array with negative numbers and zero
    x = np.array([[-1.0, 0.0], [1.0, -2.0]], dtype=np.float16)
    input_dict = {"x": x, "name": None}
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
