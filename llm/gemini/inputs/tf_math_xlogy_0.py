
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_xlogy_inputs():
    list_of_inputs = []

    # Input 1: float32, basic case
    x = np.array([1.0, 2.0, 0.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    name = "xlogy_basic"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, different values
    x = np.array([-1.0, 0.0, 3.14], dtype=np.float64)
    y = np.array([0.5, 1.0, 2.71], dtype=np.float64)
    name = "xlogy_float64"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: complex64
    x = np.array([1 + 1j, 2 - 2j, 0 + 0j], dtype=np.complex64)
    y = np.array([2 - 1j, 3 + 2j, 1 + 0j], dtype=np.complex64)
    name = "xlogy_complex64"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex128
    x = np.array([1 + 1j, 2 - 2j, 3 + 3j], dtype=np.complex128)
    y = np.array([2 - 1j, 3 + 2j, 4 - 3j], dtype=np.complex128)
    name = "xlogy_complex128"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: bfloat16
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32).astype(np.float16)
    y = np.array([2.0, 3.0, 4.0], dtype=np.float32).astype(np.float16)
    name = "xlogy_bfloat16"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: half
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32).astype(np.float16)
    y = np.array([2.0, 3.0, 4.0], dtype=np.float32).astype(np.float16)
    name = "xlogy_half"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, multi-dimensional
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    name = "xlogy_multi_dim"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: float32, all zeros in x
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    name = "xlogy_zeros"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, y has ones
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    name = "xlogy_ones"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, x and y are the same
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = "xlogy_same"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.xlogy"] = tf_math_xlogy_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.xlogy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.xlogy'.")

check_valid('tf.math.xlogy', generated_inputs['tf.math.xlogy'], lib="tf", suffix=0)
