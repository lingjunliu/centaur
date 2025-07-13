
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
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    x = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "reciprocal_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: complex64, 1D array
    x = np.array([1 + 1j, 2 - 2j, 3 + 0j], dtype=np.complex64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int32, 1D array
    x = np.array([1, 2, 3, -4], dtype=np.int32)
    input_dict = {"x": x, "name": "reciprocal_int"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: bfloat16, 2D array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16).astype(tf.bfloat16.as_numpy_dtype)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, 2D array
    x = np.array([[1 + 1j, 2 - 2j], [3 + 0j, 4 - 1j]], dtype=np.complex128)
    input_dict = {"x": x, "name": "reciprocal_complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int8, 1D array
    x = np.array([1, -2, 3, -4], dtype=np.int8)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64, 3D array
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    input_dict = {"x": x, "name": "reciprocal_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half, 1D array
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int16, 2D array with negative values
    x = np.array([[-1, 2], [-3, 4]], dtype=np.int16)
    input_dict = {"x": x, "name": "reciprocal_int16"}
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
