
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_floordiv_inputs():
    list_of_inputs = []

    # Input 1: Basic integer division
    x = tf.constant(np.array([10, 11, 12]), dtype=tf.int32)
    y = tf.constant(np.array([3, 3, 3]), dtype=tf.int32)
    name = "floordiv_int"
    input_dict = {"x": x.numpy(), "y": y.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative numbers
    x = tf.constant(np.array([-10, -11, -12]), dtype=tf.int32)
    y = tf.constant(np.array([3, 3, 3]), dtype=tf.int32)
    name = "floordiv_negative"
    input_dict = {"x": x.numpy(), "y": y.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Mixed positive and negative
    x = tf.constant(np.array([-10, 11, -12]), dtype=tf.int32)
    y = tf.constant(np.array([3, -3, 3]), dtype=tf.int32)
    name = "floordiv_mixed"
    input_dict = {"x": x.numpy(), "y": y.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Floating-point numbers
    x = tf.constant(np.array([10.5, 11.2, 12.7]), dtype=tf.float32)
    y = tf.constant(np.array([3.0, 3.0, 3.0]), dtype=tf.float32)
    name = "floordiv_float"
    input_dict = {"x": x.numpy(), "y": y.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Floating-point with negative numbers
    x = tf.constant(np.array([-10.5, 11.2, -12.7]), dtype=tf.float32)
    y = tf.constant(np.array([3.0, -3.0, 3.0]), dtype=tf.float32)
    name = "floordiv_float_negative"
    input_dict = {"x": x.numpy(), "y": y.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.floordiv"] = tf_math_floordiv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.floordiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.floordiv'.")

check_valid('tf.math.floordiv', generated_inputs['tf.math.floordiv'], lib="tf", suffix=0)
