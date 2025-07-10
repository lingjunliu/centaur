
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_divmod_inputs():
    list_of_inputs = []

    # Input 1: Basic integer division
    x1 = tf.constant(np.array([10, 7, 4]), dtype=tf.int32)
    x2 = tf.constant(np.array([3, 3, 3]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Division with negative numbers
    x1 = tf.constant(np.array([-10, 7, -4]), dtype=tf.int32)
    x2 = tf.constant(np.array([3, -3, 3]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Floating-point division
    x1 = tf.constant(np.array([10.0, 7.0, 4.0]), dtype=tf.float32)
    x2 = tf.constant(np.array([3.0, 3.0, 3.0]), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting
    x1 = tf.constant(np.array([10, 7, 4]), dtype=tf.int32)
    x2 = tf.constant(3, dtype=tf.int32)
    x2 = tf.cast(x2, dtype=x1.dtype)
    x2 = tf.reshape(x2, ())

    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional arrays
    x1 = tf.constant(np.array([[10, 7], [4, 11]]), dtype=tf.int32)
    x2 = tf.constant(np.array([[3, 2], [2, 5]]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different dtypes that can be promoted to each other
    x1 = tf.constant(np.array([10, 7, 4]), dtype=tf.int64)
    x2 = tf.constant(np.array([3, 3, 3]), dtype=tf.int32)
    x2 = tf.cast(x2, dtype=x1.dtype) # Ensure same dtype to avoid failure
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger numbers
    x1 = tf.constant(np.array([10000, 7000, 4000]), dtype=tf.int32)
    x2 = tf.constant(np.array([300, 300, 300]), dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Mixed positive and negative, broadcasting
    x1 = tf.constant(np.array([-10, 7, -4]), dtype=tf.int32)
    x2 = tf.constant(-3, dtype=tf.int32)
    x2 = tf.cast(x2, dtype=x1.dtype)
    x2 = tf.reshape(x2, ())
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: different shapes with broadcasting
    x1 = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]), dtype=tf.int32)
    x2 = tf.constant(np.array([1, 2, 3]), dtype=tf.int32)

    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: with float64

    x1 = tf.constant(np.array([10.0, 7.0, 4.0]), dtype=tf.float64)
    x2 = tf.constant(np.array([3.0, 3.0, 3.0]), dtype=tf.float64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.divmod"] = tf_experimental_numpy_divmod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.divmod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.divmod'.")

check_valid('tf.experimental.numpy.divmod', generated_inputs['tf.experimental.numpy.divmod'], lib="tf", suffix=0)
