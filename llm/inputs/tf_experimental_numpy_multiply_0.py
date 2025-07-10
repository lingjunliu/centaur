
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_multiply_inputs():
    list_of_inputs = []

    # Input 1: Basic integer tensors
    x1 = tf.constant(np.array([1, 2, 3]), dtype=tf.int32)
    x2 = tf.constant(np.array([4, 5, 6]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic float tensors
    x1 = tf.constant(np.array([1.0, 2.0, 3.0]), dtype=tf.float32)
    x2 = tf.constant(np.array([4.0, 5.0, 6.0]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional integer tensors
    x1 = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32)
    x2 = tf.constant(np.array([[5, 6], [7, 8]]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional float tensors
    x1 = tf.constant(np.array([[1.5, 2.5], [3.5, 4.5]]), dtype=tf.float32)
    x2 = tf.constant(np.array([[5.5, 6.5], [7.5, 8.5]]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative integer values
    x1 = tf.constant(np.array([-1, -2, -3]), dtype=tf.int32)
    x2 = tf.constant(np.array([4, 5, 6]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative float values
    x1 = tf.constant(np.array([-1.0, -2.0, -3.0]), dtype=tf.float32)
    x2 = tf.constant(np.array([4.0, 5.0, 6.0]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero values
    x1 = tf.constant(np.array([0, 1, 2]), dtype=tf.int32)
    x2 = tf.constant(np.array([3, 4, 0]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different shapes (but broadcastable)
    x1 = tf.constant(np.array([1, 2, 3]), dtype=tf.int32)
    x2 = tf.constant(np.array(2), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different shapes (but broadcastable) with float
    x1 = tf.constant(np.array([[1.0, 2.0]]), dtype=tf.float32)
    x2 = tf.constant(np.array([[3.0], [4.0]]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger tensors
    x1 = tf.constant(np.random.rand(100, 100).astype(np.float32))
    x2 = tf.constant(np.random.rand(100, 100).astype(np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.multiply"] = tf_experimental_numpy_multiply_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.multiply' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.multiply'.")

check_valid('tf.experimental.numpy.multiply', generated_inputs['tf.experimental.numpy.multiply'], lib="tf", suffix=0)
