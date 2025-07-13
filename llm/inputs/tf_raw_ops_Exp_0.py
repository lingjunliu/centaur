
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_exp_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    x = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64), "name": "exp_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bfloat16, scalar
    x = np.array(0.5, dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.bfloat16), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half, 3D array
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float16), "name": "another_exp"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, 1D array
    x = np.array([1 + 1j, 2 - 2j, 3 + 0j], dtype=np.complex64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.complex64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, scalar
    x = np.array(1j, dtype=np.complex128)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.complex128), "name": "complex_exp"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, empty array
    x = np.array([], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, large values
    x = np.array([100.0, -100.0], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": "large_exp"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, multi-dimensional array with a different shape
    x = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128, 2D array
    x = np.array([[1+1j, 2-2j], [3+0j, 0+4j]], dtype=np.complex128)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.complex128), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Exp"] = tf_raw_ops_exp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Exp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Exp'.")

check_valid('tf.raw_ops.Exp', generated_inputs['tf.raw_ops.Exp'], lib="tf", suffix=0)
