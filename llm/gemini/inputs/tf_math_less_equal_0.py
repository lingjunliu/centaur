
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_less_equal_inputs():
    list_of_inputs = []

    # Input 1: Basic case with integers
    x = tf.constant(np.array([1, 2, 3])).numpy()
    y = tf.constant(np.array([2, 2, 2])).numpy()
    name = "less_equal_1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic case with floats
    x = tf.constant(np.array([1.0, 2.0, 3.0])).numpy()
    y = tf.constant(np.array([2.0, 1.0, 4.0])).numpy()
    name = "less_equal_2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcasting with integers
    x = tf.constant(np.array([1, 2, 3])).numpy()
    y = tf.constant(np.array([2])).numpy()
    name = "less_equal_3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting with floats
    x = tf.constant(np.array([1.0, 2.0, 3.0])).numpy()
    y = tf.constant(np.array([2.0])).numpy()
    name = "less_equal_4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional arrays with integers
    x = tf.constant(np.array([[1, 2], [3, 4]])).numpy()
    y = tf.constant(np.array([[2, 1], [4, 3]])).numpy()
    name = "less_equal_5"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-dimensional arrays with floats
    x = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]])).numpy()
    y = tf.constant(np.array([[2.0, 1.0], [4.0, 3.0]])).numpy()
    name = "less_equal_6"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative values with integers
    x = tf.constant(np.array([-1, -2, -3])).numpy()
    y = tf.constant(np.array([0, -1, -4])).numpy()
    name = "less_equal_7"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative values with floats
    x = tf.constant(np.array([-1.0, -2.0, -3.0])).numpy()
    y = tf.constant(np.array([0.0, -1.0, -4.0])).numpy()
    name = "less_equal_8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All equal values
    x = tf.constant(np.array([5, 5, 5])).numpy()
    y = tf.constant(np.array([5, 5, 5])).numpy()
    name = "less_equal_9"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different shapes with broadcasting
    x = tf.constant(np.array([[1, 2, 3], [4, 5, 6]])).numpy()
    y = tf.constant(np.array([2, 5, 7])).numpy()
    name = "less_equal_10"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.less_equal"] = tf_math_less_equal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.less_equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.less_equal'.")

check_valid('tf.math.less_equal', generated_inputs['tf.math.less_equal'], lib="tf", suffix=0)
