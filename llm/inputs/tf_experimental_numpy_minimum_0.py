
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_minimum_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive integers
    x1 = tf.constant(np.array([1, 2, 3, 4]))
    x2 = tf.constant(np.array([4, 3, 2, 1]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With negative integers
    x1 = tf.constant(np.array([-1, -2, 3, -4]))
    x2 = tf.constant(np.array([4, -3, -2, 1]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With floating-point numbers
    x1 = tf.constant(np.array([1.5, 2.7, 3.0, 4.2]))
    x2 = tf.constant(np.array([4.1, 3.2, 2.9, 1.0]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays
    x1 = tf.constant(np.array([[1, 2], [3, 4]]))
    x2 = tf.constant(np.array([[4, 3], [2, 1]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D arrays with floats
    x1 = tf.constant(np.array([[1.1, 2.2], [3.3, 4.4]]))
    x2 = tf.constant(np.array([[4.4, 3.3], [2.2, 1.1]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D arrays
    x1 = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    x2 = tf.constant(np.array([[[8, 7], [6, 5]], [[4, 3], [2, 1]]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting example
    x1 = tf.constant(np.array([1, 2, 3]))
    x2 = tf.constant(np.array(2))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All negative values
    x1 = tf.constant(np.array([-1, -2, -3, -4]))
    x2 = tf.constant(np.array([-4, -3, -2, -1]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Mixed positive and negative floats
    x1 = tf.constant(np.array([-1.5, 2.7, -3.0, 4.2]))
    x2 = tf.constant(np.array([4.1, -3.2, 2.9, -1.0]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Scalar tensors
    x1 = tf.constant(np.array([5]))
    x2 = tf.constant(np.array([2]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.minimum"] = tf_experimental_numpy_minimum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.minimum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.minimum'.")

check_valid('tf.experimental.numpy.minimum', generated_inputs['tf.experimental.numpy.minimum'], lib="tf", suffix=0)
