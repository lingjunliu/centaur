
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_less_inputs():
    list_of_inputs = []

    # Input 1: Basic integer comparison
    x = tf.constant(np.array([1, 2, 3]), dtype=tf.int32)
    y = tf.constant(np.array([2, 2, 2]), dtype=tf.int32)
    name = "less_basic_int"
    input_dict = {"x": x.numpy(), "y": y.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float comparison with broadcasting
    x = tf.constant(np.array([1.0, 2.0, 3.0]), dtype=tf.float32)
    y = tf.constant(np.array([2.0]), dtype=tf.float32)
    name = "less_float_broadcast"
    input_dict = {"x": x.numpy(), "y": y.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative integers
    x = tf.constant(np.array([-1, -2, -3]), dtype=tf.int32)
    y = tf.constant(np.array([0, -1, -4]), dtype=tf.int32)
    name = "less_negative_int"
    input_dict = {"x": x.numpy(), "y": y.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.less"] = tf_math_less_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.less' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.less'.")

check_valid('tf.math.less', generated_inputs['tf.math.less'], lib="tf", suffix=0)
