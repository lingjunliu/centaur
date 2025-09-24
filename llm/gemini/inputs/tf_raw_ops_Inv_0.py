
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_inv_inputs():
    list_of_inputs = []

    # Input 1: float32 scalar
    x = np.array(2.0, dtype=np.float32)
    input_dict = {"x": x, "name": "scalar_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 vector
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "name": "vector_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32 matrix
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "matrix_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64
    x = np.array(5.0, dtype=np.float64)
    input_dict = {"x": x, "name": "scalar_float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64
    x = np.array(1 + 1j, dtype=np.complex64)
    input_dict = {"x": x, "name": "complex64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128
    x = np.array(2 + 2j, dtype=np.complex128)
    input_dict = {"x": x, "name": "complex128"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: half (float16)
    x = np.array(3.0, dtype=np.float16)
    input_dict = {"x": x, "name": "float16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: negative float32
    x = np.array(-2.0, dtype=np.float32)
    input_dict = {"x": x, "name": "negative_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: zeros float32
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"x": x, "name": "zeros_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Inv"] = tf_raw_ops_inv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Inv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Inv'.")

check_valid('tf.raw_ops.Inv', generated_inputs['tf.raw_ops.Inv'], lib="tf", suffix=0)
