
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_lgamma_inputs():
    list_of_inputs = []

    # Input 1: Basic positive float32
    x = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic positive float64
    x = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Basic negative float32
    x = tf.constant(np.array([-1.0, -2.0, -3.0], dtype=np.float32))
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Basic negative float64
    x = tf.constant(np.array([-1.0, -2.0, -3.0], dtype=np.float64))
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero float32
    x = tf.constant(np.array([0.0], dtype=np.float32))
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zero float64
    x = tf.constant(np.array([0.0], dtype=np.float64))
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Mixed positive and negative float32
    x = tf.constant(np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32))
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.lgamma"] = tf_math_lgamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.lgamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.lgamma'.")

check_valid('tf.math.lgamma', generated_inputs['tf.math.lgamma'], lib="tf", suffix=0)
