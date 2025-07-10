
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_greater_equal_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([5, 4, 6, 7], dtype=np.int32)
    y = np.array([5, 2, 5, 10], dtype=np.int32)
    input_dict = {"x": tf.constant(x).numpy(), "y": tf.constant(y).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([5, 4, 6, 7], dtype=np.int32)
    y = np.array([5], dtype=np.int32)
    input_dict = {"x": tf.constant(x).numpy(), "y": tf.constant(y).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.5, 1.5, 1.5], dtype=np.float32)
    input_dict = {"x": tf.constant(x).numpy(), "y": tf.constant(y).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([-1, -2, -3], dtype=np.int32)
    y = np.array([0, -1, -4], dtype=np.int32)
    input_dict = {"x": tf.constant(x).numpy(), "y": tf.constant(y).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[2, 1], [4, 3]], dtype=np.int32)
    input_dict = {"x": tf.constant(x).numpy(), "y": tf.constant(y).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    y = np.array([[[2, 1], [4, 3]], [[6, 5], [8, 7]]], dtype=np.float32)
    input_dict = {"x": tf.constant(x).numpy(), "y": tf.constant(y).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([1, 2, 3], dtype=np.int64)
    input_dict = {"x": tf.constant(x).numpy(), "y": tf.constant(y).numpy(), "name": "ge_test"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([5, 4, 6, 7], dtype=np.uint8)
    y = np.array([5, 2, 5, 10], dtype=np.uint8)
    input_dict = {"x": tf.constant(x).numpy(), "y": tf.constant(y).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    x = np.array([5.0, 4.0, 6.0, 7.0], dtype=np.float64)
    y = np.array([5.0, 2.0, 5.0, 10.0], dtype=np.float64)
    input_dict = {"x": tf.constant(x).numpy(), "y": tf.constant(y).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    y = np.array(3, dtype=np.int32)
    input_dict = {"x": tf.constant(x).numpy(), "y": tf.constant(y).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.greater_equal"] = tf_math_greater_equal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.greater_equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.greater_equal'.")

check_valid('tf.math.greater_equal', generated_inputs['tf.math.greater_equal'], lib="tf", suffix=0)
