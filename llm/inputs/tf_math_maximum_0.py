
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_maximum_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensors
    x = tf.constant(np.array([1.0, 2.0, 3.0]), dtype=tf.float32)
    y = tf.constant(np.array([2.0, 1.0, 4.0]), dtype=tf.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative values, int32
    x = tf.constant(np.array([-1, -2, -3]), dtype=tf.int32)
    y = tf.constant(np.array([-2, -1, -4]), dtype=tf.int32)
    input_dict = {"x": x, "y": y, "name": "max_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different shapes (broadcast), float64
    x = tf.constant(np.array([1.0, 2.0, 3.0]), dtype=tf.float64)
    y = tf.constant(np.array([2.0]), dtype=tf.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D tensors, int16
    x = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int16)
    y = tf.constant(np.array([[2, 1], [4, 3]]), dtype=tf.int16)
    input_dict = {"x": x, "y": y, "name": "max_2d_int16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero values, float32
    x = tf.constant(np.array([0.0, 0.0, 0.0]), dtype=tf.float32)
    y = tf.constant(np.array([-1.0, 0.0, 1.0]), dtype=tf.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int64
    x = tf.constant(np.array([1, 2, 3], dtype=np.int64), dtype=tf.int64)
    y = tf.constant(np.array([2, 1, 4], dtype=np.int64), dtype=tf.int64)
    input_dict = {"x": x, "y": y, "name": "max_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint8
    x = tf.constant(np.array([1, 2, 3], dtype=np.uint8), dtype=tf.uint8)
    y = tf.constant(np.array([2, 1, 4], dtype=np.uint8), dtype=tf.uint8)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16
    x = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float16), dtype=tf.float16)
    y = tf.constant(np.array([2.0, 1.0, 4.0], dtype=np.float16), dtype=tf.float16)
    input_dict = {"x": x, "y": y, "name": "max_float16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_math_maximum_inputs()
for i in range(len(inputs)):
  inputs[i]['x'] = inputs[i]['x'].numpy()
  inputs[i]['y'] = inputs[i]['y'].numpy()
generated_inputs["tf.math.maximum"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.maximum'.")

check_valid('tf.math.maximum', generated_inputs['tf.math.maximum'], lib="tf", suffix=0)
