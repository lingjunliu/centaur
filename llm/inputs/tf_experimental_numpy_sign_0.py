
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_sign_inputs():
    list_of_inputs = []
    tf.experimental.numpy.experimental_enable_numpy_behavior()

    # Input 1: Basic case with positive and negative numbers
    x = tf.constant(np.array([-1, 0, 1, -2, 2]))
    out = tf.zeros_like(x)
    where = tf.constant(True, dtype=tf.bool)

    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: All positive numbers
    x = tf.constant(np.array([1, 2, 3, 4, 5]))
    out = tf.zeros_like(x)
    where = tf.constant(True, dtype=tf.bool)

    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: All negative numbers
    x = tf.constant(np.array([-1, -2, -3, -4, -5]))
    out = tf.zeros_like(x)
    where = tf.constant(True, dtype=tf.bool)

    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: All zeros
    x = tf.constant(np.array([0, 0, 0, 0, 0]))
    out = tf.zeros_like(x)
    where = tf.constant(True, dtype=tf.bool)

    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array
    x = tf.constant(np.array([[-1, 0, 1], [-2, 0, 2]]))
    out = tf.zeros_like(x)
    where = tf.constant(True, dtype=tf.bool)

    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array
    x = tf.constant(np.array([[[1, -1], [0, 2]], [[-2, 1], [0, -1]]]))
    out = tf.zeros_like(x)
    where = tf.constant(True, dtype=tf.bool)

    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: where is false for some elements
    x = tf.constant(np.array([-1, 0, 1, -2, 2]))
    out = tf.zeros_like(x)
    where = tf.constant([True, False, True, False, True], dtype=tf.bool)

    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: x is float
    x = tf.constant(np.array([-1.5, 0.0, 1.5, -2.5, 2.5]))
    out = tf.zeros_like(x)
    where = tf.constant(True, dtype=tf.bool)

    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: where is a 2D boolean tensor
    x = tf.constant(np.array([[-1, 0], [1, -2]]))
    out = tf.zeros_like(x)
    where = tf.constant([[True, False], [False, True]], dtype=tf.bool)

    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: out is initialized with non-zero values
    x = tf.constant(np.array([-1, 0, 1]))
    out = tf.constant(np.array([1, 2, 3]), dtype=x.dtype)
    where = tf.constant(True, dtype=tf.bool)

    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: where is a scalar False
    x = tf.constant(np.array([-1, 0, 1, -2, 2]))
    out = tf.zeros_like(x)
    where = tf.constant(False, dtype=tf.bool)

    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: x is an empty tensor, but out is not empty
    x = tf.constant(np.array([]), dtype=tf.int32)
    out = tf.constant(np.array([1, 2, 3]), dtype=tf.int32)
    where = tf.constant(False, dtype=tf.bool)

    input_dict = {"x": x, "out": out, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.sign"] = tf_experimental_numpy_sign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.sign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.sign'.")

check_valid('tf.experimental.numpy.sign', generated_inputs['tf.experimental.numpy.sign'], lib="tf", suffix=0)
