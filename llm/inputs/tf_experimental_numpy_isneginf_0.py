
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_isneginf_inputs():
    list_of_inputs = []

    # Input 1: Scalar negative infinity
    x = tf.constant(np.NINF, dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar positive infinity
    x = tf.constant(np.inf, dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalar zero
    x = tf.constant(0.0, dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array with negative infinity
    x = tf.constant([np.NINF, 1.0, 2.0], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array with negative infinity
    x = tf.constant([[np.NINF, 1.0], [2.0, np.NINF]], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array with negative infinity
    x = tf.constant([[[np.NINF, 1.0], [2.0, 3.0]], [[4.0, 5.0], [np.NINF, 7.0]]], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array with mixed values including NaN
    x = tf.constant([np.NINF, np.inf, np.nan, 0.0, -np.inf], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  All negative infinity
    x = tf.constant([np.NINF, np.NINF, np.NINF], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: tf.float64
    x = tf.constant(np.array([-np.inf, 1.0, 2.0]), dtype=tf.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: tf.complex64 (should still work due to type casting)
    x = tf.constant(np.array([-np.inf, 1.0, 2.0]), dtype=tf.float32) #complex is not supported, it becomes float, -inf stays -inf
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.isneginf"] = tf_experimental_numpy_isneginf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.isneginf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.isneginf'.")

check_valid('tf.experimental.numpy.isneginf', generated_inputs['tf.experimental.numpy.isneginf'], lib="tf", suffix=0)
