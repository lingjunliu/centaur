
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_equal_inputs():
    list_of_inputs = []

    # Input 1: Basic equality with integers
    x1 = tf.constant(np.array([1, 2, 3]))
    x2 = tf.constant(np.array([1, 4, 3]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Equality with floats
    x1 = tf.constant(np.array([1.0, 2.5, 3.2]))
    x2 = tf.constant(np.array([1.0, 2.5, 3.0]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Equality with booleans
    x1 = tf.constant(np.array([True, False, True]))
    x2 = tf.constant(np.array([True, True, False]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Equality with strings (byte strings)
    x1 = tf.constant(np.array([b"a", b"b", b"c"]))
    x2 = tf.constant(np.array([b"a", b"b", b"d"]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Equality with multi-dimensional arrays
    x1 = tf.constant(np.array([[1, 2], [3, 4]]))
    x2 = tf.constant(np.array([[1, 5], [3, 4]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shapes (broadcasting)
    x1 = tf.constant(np.array([1, 2, 3]))
    x2 = tf.constant(np.array(2))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different dtypes
    x1 = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    x2 = tf.constant(np.array([1, 2, 3], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative values
    x1 = tf.constant(np.array([-1, -2, -3]))
    x2 = tf.constant(np.array([-1, -2, -4]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex numbers
    x1 = tf.constant(np.array([1+1j, 2+2j, 3+3j]))
    x2 = tf.constant(np.array([1+1j, 2+2j, 4+4j]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty arrays (using numpy empty array)
    x1 = tf.constant(np.empty((0,)))
    x2 = tf.constant(np.empty((0,)))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.equal"] = tf_experimental_numpy_equal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.equal'.")

check_valid('tf.experimental.numpy.equal', generated_inputs['tf.experimental.numpy.equal'], lib="tf", suffix=0)
