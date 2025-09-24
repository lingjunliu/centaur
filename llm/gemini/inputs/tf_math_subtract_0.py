
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_subtract_inputs():
    list_of_inputs = []

    # Input 1: Basic subtraction of two scalars
    x = tf.constant(5, dtype=tf.int32).numpy()
    y = tf.constant(2, dtype=tf.int32).numpy()
    name = "basic_subtract"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Subtraction of two 1D tensors
    x = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32).numpy()
    y = tf.constant([5, 4, 3, 2, 1], dtype=tf.float32).numpy()
    name = "1d_subtract"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Subtraction of a 2D tensor with broadcasting
    x = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int64).numpy()
    y = tf.constant([1, 2, 3], dtype=tf.int64).numpy()
    name = "2d_broadcast_subtract"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Subtraction of two 2D tensors
    x = tf.constant([[1, 2], [3, 4]], dtype=tf.float64).numpy()
    y = tf.constant([[5, 6], [7, 8]], dtype=tf.float64).numpy()
    name = "2d_subtract"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Subtraction with negative values
    x = tf.constant([-1, -2, -3], dtype=tf.int16).numpy()
    y = tf.constant([1, 2, 3], dtype=tf.int16).numpy()
    name = "negative_subtract"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex number subtraction
    x = tf.constant([1 + 1j, 2 + 2j], dtype=tf.complex64).numpy()
    y = tf.constant([3 + 3j, 4 + 4j], dtype=tf.complex64).numpy()
    name = "complex_subtract"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D Tensor subtraction
    x = np.random.rand(2, 3, 4).astype(np.float32)
    y = np.random.rand(2, 3, 4).astype(np.float32)
    name = "3d_subtract"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting with 3D and 1D tensors
    x = np.random.rand(2, 3, 4).astype(np.float64)
    y = np.random.rand(4).astype(np.float64)
    name = "3d_broadcast_subtract"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Subtraction with uint8 tensors
    x = tf.constant([10, 20, 30], dtype=tf.uint8).numpy()
    y = tf.constant([5, 10, 15], dtype=tf.uint8).numpy()
    name = "uint8_subtract"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.subtract"] = tf_math_subtract_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.subtract' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.subtract'.")

check_valid('tf.math.subtract', generated_inputs['tf.math.subtract'], lib="tf", suffix=0)
