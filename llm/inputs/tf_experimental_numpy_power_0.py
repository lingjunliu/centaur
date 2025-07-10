
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_power_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive integers
    x1 = tf.constant(np.array([1, 2, 3]), dtype=tf.int32)
    x2 = tf.constant(np.array([2, 3, 4]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Floats
    x1 = tf.constant(np.array([1.0, 2.0, 3.0]), dtype=tf.float32)
    x2 = tf.constant(np.array([2.0, 0.5, 1.0]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative numbers
    x1 = tf.constant(np.array([-1, -2, -3]), dtype=tf.int32)
    x2 = tf.constant(np.array([2, 3, 4]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different shapes (x2 is a scalar)
    x1 = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32)
    x2 = tf.constant(2, dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D arrays
    x1 = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.float32)
    x2 = tf.constant(np.array([[2, 0.5], [1, 2]]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large numbers
    x1 = tf.constant(np.array([100, 200, 300]), dtype=tf.int64)
    x2 = tf.constant(np.array([2, 3, 4]), dtype=tf.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero values
    x1 = tf.constant(np.array([0, 1, 2]), dtype=tf.int32)
    x2 = tf.constant(np.array([2, 3, 0]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting
    x1 = tf.constant(np.array([1, 2, 3]), dtype=tf.int32)
    x2 = tf.constant(np.array(2), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D tensors
    x1 = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), dtype=tf.float32)
    x2 = tf.constant(np.array([[[2, 0.5], [1, 2]], [[1, 2], [0.5, 1]]]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed types (int32 and float32)
    x1 = tf.constant(np.array([1, 2, 3]), dtype=tf.int32)
    x2 = tf.constant(np.array([2.0, 0.5, 1.0]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.power"] = tf_experimental_numpy_power_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.power' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.power'.")

check_valid('tf.experimental.numpy.power', generated_inputs['tf.experimental.numpy.power'], lib="tf", suffix=0)
