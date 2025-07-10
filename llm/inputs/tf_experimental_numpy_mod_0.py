
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_mod_inputs():
    list_of_inputs = []

    # Input 1: Basic integers
    x1 = tf.constant(np.array([10, 12, 15]), dtype=tf.int32)
    x2 = tf.constant(np.array([3, 4, 5]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative numbers
    x1 = tf.constant(np.array([-10, -12, -15]), dtype=tf.int32)
    x2 = tf.constant(np.array([3, -4, 5]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Floats
    x1 = tf.constant(np.array([10.5, 12.7, 15.2]), dtype=tf.float32)
    x2 = tf.constant(np.array([3.1, 4.2, 5.3]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting x2 as a scalar
    x1 = tf.constant(np.array([10, 12, 15]), dtype=tf.int32)
    x2 = tf.constant(3, dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D tensors
    x1 = tf.constant(np.array([[10, 12], [15, 17]]), dtype=tf.int32)
    x2 = tf.constant(np.array([[3, 4], [5, 6]]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different dtypes
    x1 = tf.constant(np.array([10, 12, 15]), dtype=tf.int64)
    x2 = tf.constant(np.array([3, 4, 5]), dtype=tf.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large numbers
    x1 = tf.constant(np.array([1000000000, 1200000000]), dtype=tf.int32)
    x2 = tf.constant(np.array([30000, 40000]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D tensors
    x1 = tf.constant(np.array([[[10, 12], [15, 17]], [[20, 22], [25, 27]]]), dtype=tf.int32)
    x2 = tf.constant(np.array([[[3, 4], [5, 6]], [[7, 8], [9, 10]]]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different shapes that broadcast
    x1 = tf.constant(np.array([[10, 12, 14], [16, 18, 20]]), dtype=tf.int32)
    x2 = tf.constant(np.array([2, 3, 4]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.mod"] = tf_experimental_numpy_mod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.mod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.mod'.")

check_valid('tf.experimental.numpy.mod', generated_inputs['tf.experimental.numpy.mod'], lib="tf", suffix=0)
