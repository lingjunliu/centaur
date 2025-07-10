
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_vdot_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive integers
    a = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    b = tf.constant(np.array([4, 5, 6], dtype=np.int32))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With negative integers
    a = tf.constant(np.array([-1, 2, -3], dtype=np.int32))
    b = tf.constant(np.array([4, -5, 6], dtype=np.int32))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With floating-point numbers
    a = tf.constant(np.array([1.5, 2.5, 3.5], dtype=np.float32))
    b = tf.constant(np.array([4.5, 5.5, 6.5], dtype=np.float32))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With a scalar
    a = tf.constant(np.array(5, dtype=np.int32))
    b = tf.constant(np.array(2, dtype=np.int32))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: With different shapes (should work due to flattening)
    a = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    b = tf.constant(np.array([5, 6, 7, 8], dtype=np.int32))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger values
    a = tf.constant(np.array([1000, 2000, 3000], dtype=np.int32))
    b = tf.constant(np.array([4000, 5000, 6000], dtype=np.int32))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multi-dimensional arrays with different shapes but compatible sizes.
    a = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    b = tf.constant(np.array([[5, 6], [7, 8]], dtype=np.int32))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single element arrays.
    a = tf.constant(np.array([5], dtype=np.int32))
    b = tf.constant(np.array([2], dtype=np.int32))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: A mix of positive and negative floats.
    a = tf.constant(np.array([-1.5, 2.5, -3.5], dtype=np.float32))
    b = tf.constant(np.array([4.5, -5.5, 6.5], dtype=np.float32))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex numbers
    a = tf.constant(np.array([1+1j, 2+2j], dtype=np.complex64))
    b = tf.constant(np.array([3+3j, 4+4j], dtype=np.complex64))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.vdot"] = tf_experimental_numpy_vdot_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.vdot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.vdot'.")

check_valid('tf.experimental.numpy.vdot', generated_inputs['tf.experimental.numpy.vdot'], lib="tf", suffix=0)
