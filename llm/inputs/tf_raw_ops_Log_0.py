
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_log_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array, positive values
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array, mixed values
    x = np.array([[0.1, 1.0], [10.0, 100.0]], dtype=np.float64)
    input_dict = {"x": tf.constant(x, dtype=tf.float64), "name": "log_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bfloat16, scalar
    x = np.array(2.718, dtype=np.float16)
    input_dict = {"x": tf.constant(x, dtype=tf.bfloat16), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half, 1D array, very small values
    x = np.array([0.001, 0.01, 0.1], dtype=np.float16)
    input_dict = {"x": tf.constant(x, dtype=tf.float16), "name": "small_log"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, 1D array
    x = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    input_dict = {"x": tf.constant(x, dtype=tf.complex64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, 2D array
    x = np.array([[1j, 1 + 1j], [2 - 2j, 2j]], dtype=np.complex128)
    input_dict = {"x": tf.constant(x, dtype=tf.complex128), "name": "complex_log"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, 3D array
    x = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {"x": tf.constant(x, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, scalar, large value
    x = np.array(1e10, dtype=np.float64)
    input_dict = {"x": tf.constant(x, dtype=tf.float64), "name": "large_log"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: bfloat16, 2D array
    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float16)
    input_dict = {"x": tf.constant(x, dtype=tf.bfloat16), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex64, scalar
    x = np.array(0.5 + 0.866j, dtype=np.complex64)
    input_dict = {"x": tf.constant(x, dtype=tf.complex64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Log"] = tf_raw_ops_log_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Log' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Log'.")

check_valid('tf.raw_ops.Log', generated_inputs['tf.raw_ops.Log'], lib="tf", suffix=0)
