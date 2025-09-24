
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_igamma_inputs():
    list_of_inputs = []

    # Input 1
    a = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    x = tf.constant(np.array([0.5, 1.5, 2.5], dtype=np.float32))
    name = "igamma_1"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    x = tf.constant(np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64))
    name = "igamma_2"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = tf.constant(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    x = tf.constant(np.array([5.0, 6.0, 7.0], dtype=np.float32))
    name = "igamma_3"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = tf.constant(np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64))
    x = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    name = "igamma_4"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = tf.constant(np.array([2.5], dtype=np.float32))
    x = tf.constant(np.array([1.0], dtype=np.float32))
    name = "igamma_5"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = tf.constant(np.array([5.0], dtype=np.float64))
    x = tf.constant(np.array([2.0], dtype=np.float64))
    name = "igamma_6"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = tf.constant(np.array([1.5, 2.5, 3.5], dtype=np.float32))
    x = tf.constant(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    name = "igamma_7"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    a = tf.constant(np.array([1.0, 1.0, 1.0], dtype=np.float64))
    x = tf.constant(np.array([1.5, 2.5, 3.5], dtype=np.float64))
    name = "igamma_8"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = tf.constant(np.array([0.5], dtype=np.float32))
    x = tf.constant(np.array([1.0], dtype=np.float32))
    name = "igamma_9"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = tf.constant(np.array([0.75], dtype=np.float64))
    x = tf.constant(np.array([0.25], dtype=np.float64))
    name = "igamma_10"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_math_igamma_inputs()

for i in range(len(inputs)):
    inputs[i]['a'] = np.array(inputs[i]['a'])
    inputs[i]['x'] = np.array(inputs[i]['x'])

generated_inputs["tf.math.igamma"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.igamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.igamma'.")

check_valid('tf.math.igamma', generated_inputs['tf.math.igamma'], lib="tf", suffix=0)
