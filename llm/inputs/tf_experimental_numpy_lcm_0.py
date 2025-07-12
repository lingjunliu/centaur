
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_lcm_inputs():
    list_of_inputs = []

    # Input 1: Basic positive integers
    x1 = tf.constant(np.array([2, 4, 6]), dtype=tf.int32)
    x2 = tf.constant(np.array([3, 6, 9]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero and positive integers
    x1 = tf.constant(np.array([0, 5, 10]), dtype=tf.int32)
    x2 = tf.constant(np.array([5, 0, 15]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: One-dimensional array
    x1 = tf.constant(np.array([7, 14, 21]), dtype=tf.int64)
    x2 = tf.constant(np.array([1, 2, 3]), dtype=tf.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Two-dimensional array
    x1 = tf.constant(np.array([[2, 4], [6, 8]]), dtype=tf.int32)
    x2 = tf.constant(np.array([[3, 6], [9, 12]]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Mixed dtypes, int64 and int32
    x1 = tf.constant(np.array([2, 4, 6]), dtype=tf.int64)
    x2 = tf.constant(np.array([3, 6, 9]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values
    x1 = tf.constant(np.array([-2, -4, -6]), dtype=tf.int32)
    x2 = tf.constant(np.array([3, 6, -9]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger values
    x1 = tf.constant(np.array([200, 400, 600]), dtype=tf.int32)
    x2 = tf.constant(np.array([300, 600, 900]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Three-dimensional array
    x1 = tf.constant(np.array([[[2, 4], [6, 8]], [[10, 12], [14, 16]]]), dtype=tf.int32)
    x2 = tf.constant(np.array([[[3, 6], [9, 12]], [[15, 18], [21, 24]]]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Mix of large and small numbers.
    x1 = tf.constant(np.array([1, 1000]), dtype=tf.int32)
    x2 = tf.constant(np.array([500, 1]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different shapes that are broadcastable.
    x1 = tf.constant(np.array([2, 4]), dtype=tf.int32)
    x2 = tf.constant(np.array(6), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.lcm"] = tf_experimental_numpy_lcm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.lcm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.lcm'.")

check_valid('tf.experimental.numpy.lcm', generated_inputs['tf.experimental.numpy.lcm'], lib="tf", suffix=0)
