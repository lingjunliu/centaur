
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_greater_inputs():
    list_of_inputs = []

    # Input 1: Basic int32 tensors
    x = tf.constant(np.array([5, 4, 6], dtype=np.int32))
    y = tf.constant(np.array([5, 2, 5], dtype=np.int32))
    name = "greater_basic_int32"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Broadcasting with int32
    x = tf.constant(np.array([5, 4, 6], dtype=np.int32))
    y = tf.constant(np.array([5], dtype=np.int32))
    name = "greater_broadcast_int32"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32 tensors
    x = tf.constant(np.array([5.0, 4.0, 6.0], dtype=np.float32))
    y = tf.constant(np.array([5.0, 2.0, 5.0], dtype=np.float32))
    name = "greater_float32"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values with float32
    x = tf.constant(np.array([-5.0, -4.0, -6.0], dtype=np.float32))
    y = tf.constant(np.array([-5.0, -2.0, -5.0], dtype=np.float32))
    name = "greater_negative_float32"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int64 tensors
    x = tf.constant(np.array([5, 4, 6], dtype=np.int64))
    y = tf.constant(np.array([5, 2, 5], dtype=np.int64))
    name = "greater_int64"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting with float64
    x = tf.constant(np.array([5.0, 4.0, 6.0], dtype=np.float64))
    y = tf.constant(np.array([5.0], dtype=np.float64))
    name = "greater_broadcast_float64"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D int32 tensors
    x = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    y = tf.constant(np.array([[2, 1], [4, 3]], dtype=np.int32))
    name = "greater_2d_int32"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 tensors with broadcasting
    x = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    y = tf.constant(np.array([2.0, 1.0], dtype=np.float32))
    name = "greater_2d_broadcast_float32"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint8 tensors
    x = tf.constant(np.array([5, 4, 6], dtype=np.uint8))
    y = tf.constant(np.array([5, 2, 5], dtype=np.uint8))
    name = "greater_uint8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Convert tensors to numpy arrays
    for input_dict in list_of_inputs:
        input_dict['x'] = input_dict['x'].numpy()
        input_dict['y'] = input_dict['y'].numpy()

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.greater"] = tf_math_greater_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.greater' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.greater'.")

check_valid('tf.math.greater', generated_inputs['tf.math.greater'], lib="tf", suffix=0)
