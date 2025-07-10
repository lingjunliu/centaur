
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_remainder_inputs():
    list_of_inputs = []

    # Input 1: Basic integers
    x1 = tf.constant(np.array([10, 12, 15]))
    x2 = tf.constant(np.array([3, 4, 5]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shapes, integer
    x1 = tf.constant(np.array([[10, 12], [15, 16]]))
    x2 = tf.constant(np.array([3, 4]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Floating-point numbers
    x1 = tf.constant(np.array([10.5, 12.3, 15.7]))
    x2 = tf.constant(np.array([3.0, 4.0, 5.0]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative numbers
    x1 = tf.constant(np.array([-10, 12, -15]))
    x2 = tf.constant(np.array([3, -4, 5]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting
    x1 = tf.constant(np.array([[10, 12], [15, 16]]))
    x2 = tf.constant(np.array([3]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger numbers
    x1 = tf.constant(np.array([1000, 2000, 3000]))
    x2 = tf.constant(np.array([300, 700, 1100]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different dtypes that can be converted
    x1 = tf.constant(np.array([10, 12, 15], dtype=np.float64))
    x2 = tf.constant(np.array([3, 4, 5], dtype=np.int32))
    x1 = tf.cast(x1, tf.float32)
    x2 = tf.cast(x2, tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: More complex shape
    x1 = tf.constant(np.random.rand(2, 3, 4))
    x2 = tf.constant(np.random.rand(4))
    x1 = tf.cast(x1, tf.float32)
    x2 = tf.cast(x2, tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All zeros
    x1 = tf.constant(np.array([0, 0, 0]))
    x2 = tf.constant(np.array([1, 2, 3]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Zeros in x2
    x1 = tf.constant(np.array([10, 12, 15]))
    x2 = tf.constant(np.array([3, 0, 5]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Rank 0 tensors
    x1 = tf.constant(10)
    x2 = tf.constant(3)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.remainder"] = tf_experimental_numpy_remainder_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.remainder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.remainder'.")

check_valid('tf.experimental.numpy.remainder', generated_inputs['tf.experimental.numpy.remainder'], lib="tf", suffix=0)
