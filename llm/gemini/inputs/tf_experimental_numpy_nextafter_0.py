
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_experimental_numpy_nextafter_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive numbers
    x1 = tf.constant(np.array([1.0, 2.0, 3.0]), dtype=tf.float32)
    x2 = tf.constant(np.array([1.1, 2.1, 3.1]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative numbers
    x1 = tf.constant(np.array([-1.0, -2.0, -3.0]), dtype=tf.float32)
    x2 = tf.constant(np.array([-0.9, -1.9, -2.9]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero values
    x1 = tf.constant(np.array([0.0, 0.0, 0.0]), dtype=tf.float32)
    x2 = tf.constant(np.array([0.1, -0.1, 0.0]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mixed positive and negative
    x1 = tf.constant(np.array([-1.0, 0.0, 1.0]), dtype=tf.float32)
    x2 = tf.constant(np.array([-0.9, 0.1, 1.1]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger numbers
    x1 = tf.constant(np.array([1e6, 2e6, 3e6]), dtype=tf.float32)
    x2 = tf.constant(np.array([1.000001e6, 2.000001e6, 3.000001e6]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Smaller numbers
    x1 = tf.constant(np.array([1e-6, 2e-6, 3e-6]), dtype=tf.float32)
    x2 = tf.constant(np.array([1.000001e-6, 2.000001e-6, 3.000001e-6]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: Different data types
    x1 = tf.constant(np.array([1.0, 2.0, 3.0]), dtype=tf.float64)
    x2 = tf.constant(np.array([1.1, 2.1, 3.1]), dtype=tf.float64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multidimensional array
    x1 = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]]), dtype=tf.float32)
    x2 = tf.constant(np.array([[1.1, 2.1], [3.1, 4.1]]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: When x1 and x2 are equal
    x1 = tf.constant(np.array([1.0, 2.0, 3.0]), dtype=tf.float32)
    x2 = tf.constant(np.array([1.0, 2.0, 3.0]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: When x1 and x2 are close, but x2 is slightly larger
    x1 = tf.constant(np.array([1.0, 2.0, 3.0]), dtype=tf.float32)
    x2 = tf.constant(np.array([1.0 + 1e-7, 2.0 + 1e-7, 3.0 + 1e-7]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.nextafter"] = tf_experimental_numpy_nextafter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.nextafter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.nextafter'.")

check_valid('tf.experimental.numpy.nextafter', generated_inputs['tf.experimental.numpy.nextafter'], lib="tf", suffix=0)
