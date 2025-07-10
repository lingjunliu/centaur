
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_reciprocal_inputs():
    list_of_inputs = []

    # Input 1: Positive floats
    x = tf.constant(np.array([1.0, 2.5, 3.7, 4.2, 5.9], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative floats
    x = tf.constant(np.array([-1.0, -2.5, -3.7, -4.2, -5.9], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Small positive floats
    x = tf.constant(np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array of floats
    x = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array of floats
    x = tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small floats
    x = tf.constant(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Mixed positive and negative numbers (floats)
    x = tf.constant(np.array([-1.0, 2.0, -3.0, 4.0, -5.0], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: More floats
    x = tf.constant(np.array([1.5, 2.7, 3.9, 4.1, 5.3], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger floats
    x = tf.constant(np.array([10.0, 20.0, 30.0], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Smaller floats
    x = tf.constant(np.array([0.01, 0.02, 0.03], dtype=np.float32))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.reciprocal"] = tf_experimental_numpy_reciprocal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.reciprocal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.reciprocal'.")

check_valid('tf.experimental.numpy.reciprocal', generated_inputs['tf.experimental.numpy.reciprocal'], lib="tf", suffix=0)
