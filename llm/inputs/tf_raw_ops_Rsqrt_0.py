
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_rsqrt_inputs():
    list_of_inputs = []

    # Input 1: float32, simple case
    x = np.array([1.0, 4.0, 9.0], dtype=np.float32)
    name = "rsqrt_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, simple case
    x = np.array([1.0, 4.0, 9.0], dtype=np.float64)
    name = "rsqrt_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bfloat16, simple case
    x = np.array([1.0, 4.0, 9.0], dtype=np.float16)
    name = "rsqrt_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half, simple case
    x = np.array([1.0, 4.0, 9.0], dtype=np.float16)
    name = "rsqrt_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, simple case
    x = np.array([1.0 + 1j, 4.0 + 2j, 9.0 + 3j], dtype=np.complex64)
    name = "rsqrt_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, simple case
    x = np.array([1.0 + 1j, 4.0 + 2j, 9.0 + 3j], dtype=np.complex128)
    name = "rsqrt_6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, multi-dimensional
    x = np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float32)
    name = "rsqrt_7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, zero value
    x = np.array([0.0], dtype=np.float64)
    name = "rsqrt_8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, larger array
    x = np.arange(1, 101, dtype=np.float32)
    name = "rsqrt_9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128, more complex numbers
    x = np.array([1.0 + 0.5j, 0.5 + 1.0j, 2.0 + 2.0j], dtype=np.complex128)
    name = "rsqrt_10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Rsqrt"] = tf_raw_ops_rsqrt_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Rsqrt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Rsqrt'.")

check_valid('tf.raw_ops.Rsqrt', generated_inputs['tf.raw_ops.Rsqrt'], lib="tf", suffix=0)
