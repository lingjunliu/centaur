
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_acosh_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 with values >= 1
    x = np.array([1.0, 1.5, 2.0, 5.0, 10.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 with some larger values
    x = np.array([1.0, 100.0, 1000.0, 10000.0], dtype=np.float64)
    input_dict = {"x": x, "name": "large_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: half (float16)
    x = np.array([1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: bfloat16
    x = np.array([1.0, 1.1, 1.2, 1.3, 1.4], dtype=tf.bfloat16.as_numpy_dtype)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64
    x = np.array([1.0 + 0j, 1.5 + 0j, 2.0 + 0j, 5.0 + 0j], dtype=np.complex64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128
    x = np.array([1.0 + 0j, 100.0 + 0j, 1000.0 + 0j], dtype=np.complex128)
    input_dict = {"x": x, "name": "complex_large"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multidimensional float32
    x = np.array([[1.0, 1.5], [2.0, 5.0]], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Higher dimension, float64
    x = np.array([[[1.0, 1.1], [1.2, 1.3]], [[1.4, 1.5], [2.0, 2.1]]], dtype=np.float64)
    input_dict = {"x": x, "name": "3d_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  float32 with 1
    x = np.array([1.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: More complex input, different name
    x = np.array([[1.0, 2.0], [5.0, 10.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "test_acosh"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
acosh_inputs = tf_raw_ops_acosh_inputs()

for i in range(len(acosh_inputs)):
    input_dict = acosh_inputs[i]
    acosh_inputs[i] = {'x': input_dict['x'], 'name': input_dict['name']}

generated_inputs["tf.raw_ops.Acosh"] = acosh_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Acosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Acosh'.")

check_valid('tf.raw_ops.Acosh', generated_inputs['tf.raw_ops.Acosh'], lib="tf", suffix=0)
