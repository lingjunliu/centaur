
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_log_inputs():
    list_of_inputs = []

    # Input 1: Simple float32 tensor
    x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64 tensor with positive values
    x = tf.constant([0.1, 1.0, 2.0], dtype=tf.float64).numpy()
    name = "log_input_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float16 tensor
    x = tf.constant([0.5, 1.0, 2.0], dtype=tf.float16).numpy()
    name = "log_input_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 tensor
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32).numpy()
    name = "log_input_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float64 tensor
    x = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=tf.float64).numpy()
    name = "log_input_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float32 with values close to zero
    x = tf.constant([1e-6, 1e-7, 1e-8], dtype=tf.float32).numpy()
    name = "log_input_6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 tensor with a large value
    x = tf.constant([1e10, 1e20, 1e30], dtype=tf.float64).numpy()
    name = "log_input_7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float32 tensor with ones
    x = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32).numpy()
    name = "log_input_8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float64 tensor with different values
    x = tf.constant([0.2, 1.5, 3.7], dtype=tf.float64).numpy()
    name = "log_input_9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Float32 tensor with negative values
    x = tf.constant([-1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    name = "log_input_10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.log"] = tf_math_log_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.log' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.log'.")

check_valid('tf.math.log', generated_inputs['tf.math.log'], lib="tf", suffix=0)
