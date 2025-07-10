
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_maximum_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays
    x1 = tf.constant(np.array([1, 5, 2, 8]))
    x2 = tf.constant(np.array([3, 2, 7, 1]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays
    x1 = tf.constant(np.array([[1, 2], [3, 4]]))
    x2 = tf.constant(np.array([[5, 6], [7, 8]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcasting
    x1 = tf.constant(np.array([1, 2, 3]))
    x2 = tf.constant(np.array(5))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values
    x1 = tf.constant(np.array([-1, -5, 2, -8]))
    x2 = tf.constant(np.array([-3, 2, -7, 1]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different dtypes
    x1 = tf.cast(tf.constant(np.array([1, 2, 3])), tf.float32)
    x2 = tf.cast(tf.constant(np.array([4, 5, 6])), tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D arrays
    x1 = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    x2 = tf.constant(np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zeros
    x1 = tf.constant(np.array([0, 0, 0]))
    x2 = tf.constant(np.array([1, -1, 0]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Mixed positive and negative and zeros
    x1 = tf.constant(np.array([-1, 0, 1]))
    x2 = tf.constant(np.array([1, -1, 0]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different shapes that can broadcast
    x1 = tf.constant(np.array([[1, 2, 3]]))
    x2 = tf.constant(np.array([4, 5, 6]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large values
    x1 = tf.constant(np.array([1e2, 2e2]))
    x2 = tf.constant(np.array([3e2, 1e2]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.maximum"] = tf_experimental_numpy_maximum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.maximum'.")

check_valid('tf.experimental.numpy.maximum', generated_inputs['tf.experimental.numpy.maximum'], lib="tf", suffix=0)
