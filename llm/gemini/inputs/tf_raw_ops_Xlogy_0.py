
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_xlogy_inputs():
    list_of_inputs = []

    # Input 1: float32, basic case
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "xlogy_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, different values
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    y = np.array([0.5, 1.0, 2.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "xlogy_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: complex64, complex values
    x = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    y = np.array([4 + 4j, 5 + 5j, 6 + 6j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": "xlogy_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex128, different complex values
    x = np.array([-1 - 1j, 0 + 0j, 1 + 1j], dtype=np.complex128)
    y = np.array([0.5 - 0.5j, 1 + 0j, 2 + 2j], dtype=np.complex128)
    input_dict = {"x": x, "y": y, "name": "xlogy_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: half, positive values
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": "xlogy_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32, multi-dimensional
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "xlogy_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64, with zeros
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y = np.array([1.0, 0.0, 2.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "xlogy_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex64, zero values
    x = np.array([0.0 + 0j, 1.0 + 1j, 2.0 + 2j], dtype=np.complex64)
    y = np.array([1.0 + 1j, 0.0 + 0j, 2.0 + 2j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": "xlogy_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex128, with negative and zero values
    x = np.array([-1.0 - 1j, 0.0 + 0j, 1.0 + 1j], dtype=np.complex128)
    y = np.array([1.0 + 1j, -1.0 -1j, 2.0 + 2j], dtype=np.complex128)
    input_dict = {"x": x, "y": y, "name": "xlogy_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: half, multi-dimensional
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    y = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": "xlogy_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Xlogy"] = tf_raw_ops_xlogy_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Xlogy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Xlogy'.")

check_valid('tf.raw_ops.Xlogy', generated_inputs['tf.raw_ops.Xlogy'], lib="tf", suffix=0)
