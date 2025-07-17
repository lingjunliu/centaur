
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sinh_inputs():
    list_of_inputs = []

    # Input 1: float32 scalar
    x = np.float32(0.0)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 1D array
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "name": "sinh_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32 2D array
    x = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "sinh_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 scalar
    x = np.float64(2.5)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 1D array with large values
    x = np.array([-10.0, 0.0, 10.0], dtype=np.float64)
    input_dict = {"x": x, "name": "sinh_large"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex64 scalar
    x = np.complex64(1.0 + 1.0j)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex64 1D array
    x = np.array([1.0 + 1.0j, 2.0 - 1.0j], dtype=np.complex64)
    input_dict = {"x": x, "name": "sinh_complex64_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex128 scalar
    x = np.complex128(2.0 - 2.0j)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: bfloat16 scalar
    x = np.array(1.5, dtype=np.float16).astype(np.float32) #Cast to float32 because numpy doesn't have bfloat16, TF converts later.
    input_dict = {"x": x, "name": "sinh_bfloat16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: half (float16) scalar
    x = np.array(0.75, dtype=np.float16)
    input_dict = {"x": x, "name": "sinh_half"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Sinh"] = tf_raw_ops_sinh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Sinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Sinh'.")

check_valid('tf.raw_ops.Sinh', generated_inputs['tf.raw_ops.Sinh'], lib="tf", suffix=0)
