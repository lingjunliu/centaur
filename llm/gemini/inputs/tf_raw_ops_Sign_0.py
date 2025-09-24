
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sign_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar
    x = np.float32(-5.0)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 1D array
    x = np.array([0.0, 2.0, -3.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 2D array
    x = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int32, 1D array
    x = np.array([0, 2, -3], dtype=np.int32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int64, scalar
    x = np.int64(10)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex64, 1D array
    x = np.array([1 + 1j, -1 - 1j, 0 + 0j], dtype=np.complex64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64, 3D array
    x = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int16, 2D array with zeros
    x = np.zeros((5, 5), dtype=np.int16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half, 1D array
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128, 2D array
    x = np.array([[1 + 2j, -2 - 1j], [0j, 3 - 4j]], dtype=np.complex128)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Sign"] = tf_raw_ops_sign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Sign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Sign'.")

check_valid('tf.raw_ops.Sign', generated_inputs['tf.raw_ops.Sign'], lib="tf", suffix=0)
