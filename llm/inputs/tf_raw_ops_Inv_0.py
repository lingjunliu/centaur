
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_inv_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([1.0, 2.0, 3.0, -4.0, 0.5], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    x = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "my_inv"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: complex64, 1D array
    x = np.array([1 + 1j, 2 - 2j, 3 + 0j], dtype=np.complex64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex128, 2D array
    x = np.array([[1 + 0j, 0 - 1j], [-1 + 1j, -1 - 0j]], dtype=np.complex128)
    input_dict = {"x": x, "name": "complex_inv"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: half (float16), scalar value
    x = np.float16(2.0)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: bfloat16, 1D array
    x = tf.cast(np.array([1.0, 2.0, -3.0]), dtype=tf.bfloat16).numpy()
    input_dict = {"x": x, "name": "bfloat16_inv"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, 3D array
    x = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, scalar
    x = np.float64(-5.0)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex64, 3D array
    x = np.random.rand(2, 2, 2).astype(np.complex64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 with inf
    x = np.array([1.0, np.inf, -1.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
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
