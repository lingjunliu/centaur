
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_float_power_inputs():
    list_of_inputs = []

    # Input 1: Basic positive floats
    x1 = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    x2 = tf.constant(np.array([2.0, 0.5, 1.0], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero values
    x1 = tf.constant(np.array([0.0, 2.0, 3.0], dtype=np.float32))
    x2 = tf.constant(np.array([2.0, 0.0, 1.0], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different shapes (matching ranks)
    x1 = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    x2 = tf.constant(np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Higher dimensions
    x1 = tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32))
    x2 = tf.constant(np.array([[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large values
    x1 = tf.constant(np.array([100.0, 200.0, 300.0], dtype=np.float32))
    x2 = tf.constant(np.array([2.0, 0.5, 1.0], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small values
    x1 = tf.constant(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    x2 = tf.constant(np.array([2.0, 0.5, 1.0], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: x1 all ones
    x1 = tf.constant(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    x2 = tf.constant(np.array([2.0, 3.0, 4.0], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Mixed positive and negative exponents
    x1 = tf.constant(np.array([4.0, 9.0, 16.0], dtype=np.float32))
    x2 = tf.constant(np.array([0.5, -0.5, 0.25], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Scalar inputs
    x1 = tf.constant(2.0, dtype=np.float32)
    x2 = tf.constant(3.0, dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger tensor with positive base and exponents
    x1 = tf.constant(np.random.rand(5, 5).astype(np.float32) + 1e-6)
    x2 = tf.constant(np.random.rand(5, 5).astype(np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.float_power"] = tf_experimental_numpy_float_power_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.float_power' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.float_power'.")

check_valid('tf.experimental.numpy.float_power', generated_inputs['tf.experimental.numpy.float_power'], lib="tf", suffix=0)
