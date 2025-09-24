
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_multiply_inputs():
    list_of_inputs = []

    # Input 1: Basic multiplication of two matrices
    x = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    y = tf.constant(np.array([[5, 6], [7, 8]], dtype=np.int32))
    name = "multiply_example_1"
    input_dict = {"x": x.numpy(), "y": y.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiplication with a scalar
    x = tf.constant(np.array([1, 2, 3], dtype=np.float32))
    y = tf.constant(np.array(2.0, dtype=np.float32))
    name = "multiply_example_2"
    input_dict = {"x": x.numpy(), "y": y.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiplication with negative values
    x = tf.constant(np.array([-1, 2, -3], dtype=np.int64))
    y = tf.constant(np.array([4, -5, 6], dtype=np.int64))
    name = "multiply_example_3"
    input_dict = {"x": x.numpy(), "y": y.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multiplication with complex numbers
    x = tf.constant(np.array([1+1j, 2-2j], dtype=np.complex64))
    y = tf.constant(np.array([3-1j, 4+2j], dtype=np.complex64))
    name = "multiply_example_4"
    input_dict = {"x": x.numpy(), "y": y.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiplication with different shapes (broadcasting)
    x = tf.constant(np.array([[1, 2]], dtype=np.float64))
    y = tf.constant(np.array([[3], [4]], dtype=np.float64))
    name = "multiply_example_5"
    input_dict = {"x": x.numpy(), "y": y.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multiplication with higher dimensions
    x = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32))
    y = tf.constant(np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32))
    name = "multiply_example_6"
    input_dict = {"x": x.numpy(), "y": y.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multiplication with uint8
    x = tf.constant(np.array([1, 2, 3], dtype=np.uint8))
    y = tf.constant(np.array([4, 5, 6], dtype=np.uint8))
    name = "multiply_example_7"
    input_dict = {"x": x.numpy(), "y": y.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multiplication with int16
    x = tf.constant(np.array([1, 2, 3], dtype=np.int16))
    y = tf.constant(np.array([4, 5, 6], dtype=np.int16))
    name = "multiply_example_9"
    input_dict = {"x": x.numpy(), "y": y.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multiplication with complex128
    x = tf.constant(np.array([1+1j, 2-2j], dtype=np.complex128))
    y = tf.constant(np.array([3-1j, 4+2j], dtype=np.complex128))
    name = "multiply_example_10"
    input_dict = {"x": x.numpy(), "y": y.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.multiply"] = tf_math_multiply_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.multiply' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.multiply'.")

check_valid('tf.math.multiply', generated_inputs['tf.math.multiply'], lib="tf", suffix=0)
