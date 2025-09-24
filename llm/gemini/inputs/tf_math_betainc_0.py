
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_betainc_inputs():
    list_of_inputs = []

    # Input 1
    a = tf.constant(np.array([0.5], dtype=np.float32))
    b = tf.constant(np.array([0.5], dtype=np.float32))
    x = tf.constant(np.array([0.5], dtype=np.float32))
    input_dict = {"a": a, "b": b, "x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = tf.constant(np.array([2.0], dtype=np.float64))
    b = tf.constant(np.array([3.0], dtype=np.float64))
    x = tf.constant(np.array([0.8], dtype=np.float64))
    input_dict = {"a": a, "b": b, "x": x, "name": "betainc_example"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = tf.constant(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    x = tf.constant(np.array([0.2, 0.5, 0.9], dtype=np.float32))
    input_dict = {"a": a, "b": b, "x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    b = tf.constant(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64))
    x = tf.constant(np.array([[0.1, 0.3], [0.6, 0.9]], dtype=np.float64))
    input_dict = {"a": a, "b": b, "x": x, "name": "betainc_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = tf.constant(np.array([0.1], dtype=np.float32))
    b = tf.constant(np.array([0.1], dtype=np.float32))
    x = tf.constant(np.array([0.99], dtype=np.float32))
    input_dict = {"a": a, "b": b, "x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = tf.constant(np.array([5.0], dtype=np.float64))
    b = tf.constant(np.array([1.0], dtype=np.float64))
    x = tf.constant(np.array([0.01], dtype=np.float64))
    input_dict = {"a": a, "b": b, "x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = tf.constant(np.array([10.0], dtype=np.float32))
    b = tf.constant(np.array([10.0], dtype=np.float32))
    x = tf.constant(np.array([0.5], dtype=np.float32))
    input_dict = {"a": a, "b": b, "x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = tf.constant(np.array([0.001], dtype=np.float64))
    b = tf.constant(np.array([0.001], dtype=np.float64))
    x = tf.constant(np.array([0.5], dtype=np.float64))
    input_dict = {"a": a, "b": b, "x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    a = tf.constant(np.array([1.5, 2.5], dtype=np.float32))
    b = tf.constant(np.array([3.5, 4.5], dtype=np.float32))
    x = tf.constant(np.array([0.7, 0.3], dtype=np.float32))
    input_dict = {"a": a, "b": b, "x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = tf.constant(np.array([2.5], dtype=np.float64))
    b = tf.constant(np.array([1.5], dtype=np.float64))
    x = tf.constant(np.array([0.9], dtype=np.float64))
    input_dict = {"a": a, "b": b, "x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
inputs = tf_math_betainc_inputs()
for input_dict in inputs:
    input_dict['a'] = input_dict['a'].numpy()
    input_dict['b'] = input_dict['b'].numpy()
    input_dict['x'] = input_dict['x'].numpy()
generated_inputs["tf.math.betainc"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.betainc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.betainc'.")

check_valid('tf.math.betainc', generated_inputs['tf.math.betainc'], lib="tf", suffix=0)
