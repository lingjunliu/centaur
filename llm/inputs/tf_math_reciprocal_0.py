
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_reciprocal_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": "reciprocal_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    x = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64)
    input_dict = {"x": tf.constant(x), "name": "reciprocal_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: complex64, 1D array
    x = np.array([1 + 1j, 2 - 2j, 3 + 0j], dtype=np.complex64)
    input_dict = {"x": tf.constant(x), "name": "reciprocal_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex128, 2D array
    x = np.array([[1 + 1j, 2 - 2j], [3 + 0j, -4j]], dtype=np.complex128)
    input_dict = {"x": tf.constant(x), "name": "reciprocal_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, 3D array
    x = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {"x": tf.constant(x), "name": "reciprocal_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, scalar
    x = np.array(5.0, dtype=np.float64)
    input_dict = {"x": tf.constant(x), "name": "reciprocal_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float16, 1D array
    x = np.array([1.0, -2.0, 3.0], dtype=np.float16)
    input_dict = {"x": tf.constant(x), "name": "reciprocal_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half, 2D array
    x = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float16)
    input_dict = {"x": tf.constant(x), "name": "reciprocal_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int32
    x = np.array([-1, 2, -3, 4], dtype=np.int32)
    input_dict = {"x": tf.cast(tf.constant(x), tf.float32), "name": "reciprocal_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int64
    x = np.array([-1, 2, -3, 4], dtype=np.int64)
    input_dict = {"x": tf.cast(tf.constant(x), tf.float64), "name": "reciprocal_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.reciprocal"] = tf_math_reciprocal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.reciprocal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.reciprocal'.")

check_valid('tf.math.reciprocal', generated_inputs['tf.math.reciprocal'], lib="tf", suffix=0)
