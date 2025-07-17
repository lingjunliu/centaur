
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_xdivy_inputs():
    list_of_inputs = []

    # Input 1: float32, basic case
    x = np.array([2.0, 4.0, 0.0], dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, different values
    x = np.array([-2.0, 0.0, 5.0, -1.0], dtype=np.float64)
    y = np.array([ 1.0, -2.0, 0.0, 0.5], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: complex64
    x = np.array([1 + 1j, 0 + 0j, 2 - 2j], dtype=np.complex64)
    y = np.array([2 + 0j, 1 + 1j, 1 - 0j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex128, zeros in x and y
    x = np.array([0 + 0j, 1 - 1j, 0 + 0j], dtype=np.complex128)
    y = np.array([1 + 1j, 0 + 0j, 2 + 2j], dtype=np.complex128)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float16, using np.float16
    x = np.array([2.0, 0.0, -4.0], dtype=np.float16)
    y = np.array([1.0, 2.0, -2.0], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32, multi-dimensional
    x = np.array([[1.0, 2.0], [0.0, 4.0]], dtype=np.float32)
    y = np.array([[0.5, 1.0], [2.0, 0.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64, multi-dimensional, negative and zero values
    x = np.array([[-1.0, 0.0], [2.0, -3.0]], dtype=np.float64)
    y = np.array([[0.5, -2.0], [0.0, 1.0]], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex64, multi-dimensional
    x = np.array([[1 + 1j, 2 - 1j], [0 + 0j, 3 + 0j]], dtype=np.complex64)
    y = np.array([[2 + 0j, 1 + 1j], [1 - 1j, 0 + 1j]], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex128, more complex values
    x = np.array([[-1 + 2j, 3 - 4j], [5 + 0j, 0 - 2j]], dtype=np.complex128)
    y = np.array([[2 - 1j, 0 + 3j], [-2 + 2j, 1 - 0j]], dtype=np.complex128)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, 3D tensor
    x = np.array([[[1.0, 2.0], [3.0, 0.0]], [[4.0, 5.0], [0.0, 6.0]]], dtype=np.float32)
    y = np.array([[[0.5, 1.0], [1.5, 2.0]], [[2.0, 2.5], [3.0, 0.0]]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Xdivy"] = tf_raw_ops_xdivy_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Xdivy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Xdivy'.")

check_valid('tf.raw_ops.Xdivy', generated_inputs['tf.raw_ops.Xdivy'], lib="tf", suffix=0)
