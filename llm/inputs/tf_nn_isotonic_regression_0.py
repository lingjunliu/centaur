
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_isotonic_regression_inputs():
    list_of_inputs = []

    # Input 1
    inputs = tf.constant([3, 1, 2], dtype=tf.float32)
    decreasing = True
    axis = 0
    input_dict = {'inputs': inputs.numpy(), 'decreasing': decreasing, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = tf.constant([[3, 1, 2], [1, 3, 4]], dtype=tf.float32)
    decreasing = True
    axis = 1
    input_dict = {'inputs': inputs.numpy(), 'decreasing': decreasing, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = tf.constant([[3, 1, 2], [1, 3, 4]], dtype=tf.float32)
    decreasing = False
    axis = 1
    input_dict = {'inputs': inputs.numpy(), 'decreasing': decreasing, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = tf.constant([[-3, -1, -2], [-1, -3, -4]], dtype=tf.float32)
    decreasing = True
    axis = 1
    input_dict = {'inputs': inputs.numpy(), 'decreasing': decreasing, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = tf.constant([[-3, -1, -2], [-1, -3, -4]], dtype=tf.float32)
    decreasing = False
    axis = 1
    input_dict = {'inputs': inputs.numpy(), 'decreasing': decreasing, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
    decreasing = True
    axis = 0
    input_dict = {'inputs': inputs.numpy(), 'decreasing': decreasing, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
    decreasing = False
    axis = 0
    input_dict = {'inputs': inputs.numpy(), 'decreasing': decreasing, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    inputs = tf.constant([5, 4, 3, 2, 1], dtype=tf.float32)
    decreasing = True
    axis = 0
    input_dict = {'inputs': inputs.numpy(), 'decreasing': decreasing, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = tf.constant([5, 4, 3, 2, 1], dtype=tf.float32)
    decreasing = False
    axis = 0
    input_dict = {'inputs': inputs.numpy(), 'decreasing': decreasing, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    inputs = tf.constant([[[1, 5], [2, 4], [3, 3]]], dtype=tf.float32)
    decreasing = True
    axis = 2
    input_dict = {'inputs': inputs.numpy(), 'decreasing': decreasing, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.isotonic_regression"] = tf_nn_isotonic_regression_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.isotonic_regression' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.isotonic_regression'.")

check_valid('tf.nn.isotonic_regression', generated_inputs['tf.nn.isotonic_regression'], lib="tf", suffix=0)
