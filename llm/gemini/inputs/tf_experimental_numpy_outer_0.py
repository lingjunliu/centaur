
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_outer_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays
    a = tf.constant(np.array([1, 2, 3]))
    b = tf.constant(np.array([4, 5, 6]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays with negative values
    a = tf.constant(np.array([-1, 0, 1]))
    b = tf.constant(np.array([-2, 2, -3]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Arrays with different dtypes
    a = tf.constant(np.array([1.0, 2.0, 3.0]))
    b = tf.constant(np.array([4, 5, 6]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: One array with a single element
    a = tf.constant(np.array([1]))
    b = tf.constant(np.array([4, 5, 6]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Both arrays with single elements
    a = tf.constant(np.array([1]))
    b = tf.constant(np.array([4]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty arrays (should still work, resulting in an empty array)
    a = tf.constant(np.array([]))
    b = tf.constant(np.array([]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: One array empty
    a = tf.constant(np.array([1, 2]))
    b = tf.constant(np.array([]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: a is 2D and b is 1D. Reducing dimensions to 1D
    a = tf.constant(np.array([1, 2]))
    b = tf.constant(np.array([5, 6]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: a is 1D and b is 2D. Reducing dimensions to 1D
    a = tf.constant(np.array([1, 2]))
    b = tf.constant(np.array([5, 6]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Both are 2D. Reducing dimensions to 1D
    a = tf.constant(np.array([1, 2]))
    b = tf.constant(np.array([5, 6]))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.outer"] = tf_experimental_numpy_outer_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.outer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.outer'.")

check_valid('tf.experimental.numpy.outer', generated_inputs['tf.experimental.numpy.outer'], lib="tf", suffix=0)
