
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_inner_inputs():
    list_of_inputs = []
    tf.experimental.numpy.experimental_enable_numpy_behavior()

    # Input 1: Basic 1D arrays
    a = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    b = tf.constant(np.array([4, 5, 6], dtype=np.int32))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic 2D arrays
    a = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    b = tf.constant(np.array([[5, 6], [7, 8]], dtype=np.int32))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different shapes that are compatible
    a = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    b = tf.constant(np.array([[4, 5, 6], [7, 8, 9]], dtype=np.int32))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float arrays
    a = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = tf.constant(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values
    a = tf.constant(np.array([-1, 2, -3], dtype=np.int32))
    b = tf.constant(np.array([4, -5, 6], dtype=np.int32))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zeros
    a = tf.constant(np.array([0, 0, 0], dtype=np.int32))
    b = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger arrays
    a = tf.constant(np.random.rand(100).astype(np.float32))
    b = tf.constant(np.random.rand(100).astype(np.float32))
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.inner"] = tf_experimental_numpy_inner_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.inner' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.inner'.")

check_valid('tf.experimental.numpy.inner', generated_inputs['tf.experimental.numpy.inner'], lib="tf", suffix=0)
