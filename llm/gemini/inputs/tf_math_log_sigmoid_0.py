
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_log_sigmoid_inputs():
    list_of_inputs = []

    # Input 1: Basic test with a single float value
    x = tf.constant(0.0, dtype=tf.float32).numpy()
    name = "input_0"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive values
    x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    name = "input_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values
    x = tf.constant([-1.0, -2.0, -3.0], dtype=tf.float32).numpy()
    name = "input_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mixed positive and negative values
    x = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float32).numpy()
    name = "input_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large positive value
    x = tf.constant(100.0, dtype=tf.float32).numpy()
    name = "input_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large negative value
    x = tf.constant(-100.0, dtype=tf.float32).numpy()
    name = "input_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D tensor
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32).numpy()
    name = "input_6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D tensor with mixed values
    x = tf.constant([[-1.0, 0.0], [1.0, -2.0]], dtype=tf.float32).numpy()
    name = "input_7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D tensor
    x = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=tf.float32).numpy()
    name = "input_8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64
    x = tf.constant(1.0, dtype=tf.float64).numpy()
    name = "input_9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.log_sigmoid"] = tf_math_log_sigmoid_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.log_sigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.log_sigmoid'.")

check_valid('tf.math.log_sigmoid', generated_inputs['tf.math.log_sigmoid'], lib="tf", suffix=0)
