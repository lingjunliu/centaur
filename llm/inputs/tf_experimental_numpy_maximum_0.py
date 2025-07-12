
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_maximum_inputs():
    list_of_inputs = []

    # Input 1: Basic positive integers
    x1 = tf.constant(np.array([1, 2, 3]))
    x2 = tf.constant(np.array([4, 0, 5]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Mixed positive and negative integers
    x1 = tf.constant(np.array([-1, 2, -3]))
    x2 = tf.constant(np.array([4, -2, 5]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Floating point numbers
    x1 = tf.constant(np.array([1.5, 2.0, 3.7]))
    x2 = tf.constant(np.array([4.2, 0.0, 5.1]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mixed integers and floats
    x1 = tf.constant(np.array([1, 2.5, 3]))
    x2 = tf.constant(np.array([4.0, 0, 5.5]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D arrays
    x1 = tf.constant(np.array([[1, 2], [3, 4]]))
    x2 = tf.constant(np.array([[4, 0], [5, 6]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D arrays with negative values
    x1 = tf.constant(np.array([[-1, 2], [-3, 4]]))
    x2 = tf.constant(np.array([[4, -2], [5, -6]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different shapes (broadcasting)
    x1 = tf.constant(np.array([1, 2, 3]))
    x2 = tf.constant(np.array(2))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different shapes (broadcasting) - more complex
    x1 = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]))
    x2 = tf.constant(np.array([0, 1, 0]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All negative values
    x1 = tf.constant(np.array([-1, -2, -3]))
    x2 = tf.constant(np.array([-4, -5, -6]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger values
    x1 = tf.constant(np.array([100, 200, 300]))
    x2 = tf.constant(np.array([50, 250, 100]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.maximum"] = tf_experimental_numpy_maximum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.maximum'.")

check_valid('tf.experimental.numpy.maximum', generated_inputs['tf.experimental.numpy.maximum'], lib="tf", suffix=0)
