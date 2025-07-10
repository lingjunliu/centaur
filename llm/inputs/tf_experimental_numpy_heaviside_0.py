
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_experimental_numpy_heaviside_inputs():
    list_of_inputs = []

    # Input 1: Basic positive values
    x1 = tf.constant(np.array([1, 2, 3], dtype=np.float32))
    x2 = tf.constant(np.array([0, 0.5, 1], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic negative values
    x1 = tf.constant(np.array([-1, -2, -3], dtype=np.float32))
    x2 = tf.constant(np.array([0, 0.5, 1], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero values
    x1 = tf.constant(np.array([0, 0, 0], dtype=np.float32))
    x2 = tf.constant(np.array([0, 0.5, 1], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mixed values
    x1 = tf.constant(np.array([-1, 0, 1], dtype=np.float32))
    x2 = tf.constant(np.array([0, 0.5, 1], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional array
    x1 = tf.constant(np.array([[-1, 0], [1, 2]], dtype=np.float32))
    x2 = tf.constant(np.array([[0.5, 1], [1.5, 2]], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger values
    x1 = tf.constant(np.array([1000, -1000, 0], dtype=np.float32))
    x2 = tf.constant(np.array([0, 1, 0.5], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different shapes
    x1 = tf.constant(np.array([1, 2], dtype=np.float32))
    x2 = tf.constant(np.array([0.5, 1.5], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All zeros
    x1 = tf.zeros((2, 3), dtype=tf.float32)
    x2 = tf.ones((2, 3), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Negative x2 values
    x1 = tf.constant([1, 2, 3], dtype=tf.float32)
    x2 = tf.constant([-1, -2, -3], dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: x1 with only one element and x2 a numpy array
    x1 = tf.constant(np.array([5.0], dtype=np.float32))
    x2 = tf.constant(np.array([0.0], dtype=np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.heaviside"] = tf_experimental_numpy_heaviside_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.heaviside' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.heaviside'.")

check_valid('tf.experimental.numpy.heaviside', generated_inputs['tf.experimental.numpy.heaviside'], lib="tf", suffix=0)
