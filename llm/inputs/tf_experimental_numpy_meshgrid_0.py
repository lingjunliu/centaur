
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_meshgrid_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two 1D tensors
    x = tf.constant(np.array([1, 2, 3]))
    y = tf.constant(np.array([4, 5]))
    input_dict = {"xi": [x, y]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Three 1D tensors
    x = tf.constant(np.array([1, 2]))
    y = tf.constant(np.array([3, 4]))
    z = tf.constant(np.array([5, 6]))
    input_dict = {"xi": [x, y, z]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tensors with different shapes
    x = tf.constant(np.array([1, 2, 3, 4]))
    y = tf.constant(np.array([5, 6]))
    input_dict = {"xi": [x, y]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values
    x = tf.constant(np.array([-1, 0, 1]))
    y = tf.constant(np.array([-2, 2]))
    input_dict = {"xi": [x, y]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float values
    x = tf.constant(np.array([1.0, 2.0]))
    y = tf.constant(np.array([3.0, 4.0]))
    input_dict = {"xi": [x, y]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: One tensor only
    x = tf.constant(np.array([1, 2, 3]))
    input_dict = {"xi": [x]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger tensors
    x = tf.constant(np.arange(10))
    y = tf.constant(np.arange(5))
    input_dict = {"xi": [x, y]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zeros
    x = tf.constant(np.array([0, 0, 0]))
    y = tf.constant(np.array([0, 0]))
    input_dict = {"xi": [x, y]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Mixed positive and negative floats
    x = tf.constant(np.array([-1.5, 0.0, 2.5]))
    y = tf.constant(np.array([-0.5, 1.5]))
    input_dict = {"xi": [x, y]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Tensors with only one element
    x = tf.constant(np.array([5]))
    y = tf.constant(np.array([10]))
    input_dict = {"xi": [x, y]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.meshgrid"] = tf_experimental_numpy_meshgrid_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.meshgrid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.meshgrid'.")

check_valid('tf.experimental.numpy.meshgrid', generated_inputs['tf.experimental.numpy.meshgrid'], lib="tf", suffix=0)
