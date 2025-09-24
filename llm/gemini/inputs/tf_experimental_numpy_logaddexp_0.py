
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.experimental.numpy.experimental_enable_numpy_behavior()

def tf_experimental_numpy_logaddexp_inputs():
    list_of_inputs = []

    # Input 1: Basic example with positive numbers
    x1 = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    x2 = tf.constant(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Example with negative numbers
    x1 = tf.constant(np.array([-1.0, -2.0, -3.0], dtype=np.float32))
    x2 = tf.constant(np.array([-4.0, -5.0, -6.0], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Example with mixed positive and negative numbers
    x1 = tf.constant(np.array([-1.0, 2.0, -3.0], dtype=np.float32))
    x2 = tf.constant(np.array([4.0, -5.0, 6.0], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Example with zeros
    x1 = tf.constant(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    x2 = tf.constant(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Example with one tensor having larger values
    x1 = tf.constant(np.array([100.0, 200.0, 300.0], dtype=np.float32))
    x2 = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Example with one tensor having smaller values
    x1 = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    x2 = tf.constant(np.array([100.0, 200.0, 300.0], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Example with 2D tensors
    x1 = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    x2 = tf.constant(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Example with 2D tensors and negative values
    x1 = tf.constant(np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32))
    x2 = tf.constant(np.array([[-5.0, -6.0], [-7.0, -8.0]], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Example with different data types
    x1 = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    x2 = tf.constant(np.array([4.0, 5.0, 6.0], dtype=np.float64))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Example with 3D tensors
    x1 = tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32))
    x2 = tf.constant(np.array([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.logaddexp"] = tf_experimental_numpy_logaddexp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.logaddexp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.logaddexp'.")

check_valid('tf.experimental.numpy.logaddexp', generated_inputs['tf.experimental.numpy.logaddexp'], lib="tf", suffix=0)
