
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_rot90_inputs():
    list_of_inputs = []

    # Input 1
    m = tf.constant([[1, 2], [3, 4]]).numpy()
    k = 1
    axes = (0, 1)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    m = tf.constant([[1, 2], [3, 4]]).numpy()
    k = 2
    axes = (0, 1)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    m = tf.constant([[1, 2], [3, 4]]).numpy()
    k = 3
    axes = (0, 1)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    m = tf.constant([[1, 2], [3, 4]]).numpy()
    k = -1
    axes = (0, 1)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    m = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    k = 1
    axes = (0, 1)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    m = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    k = 2
    axes = (1, 2)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    m = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    k = -1
    axes = (0, 2)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    m = tf.constant([1, 2, 3]).numpy()
    k = 1
    axes = (0, 0) # changed from (0,) to (0,0)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    m = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]).numpy()
    k = 1
    axes = (0, 2)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    m = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]).numpy()
    k = 2
    axes = (0, 1)
    input_dict = {"m": m, "k": k, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.rot90"] = tf_experimental_numpy_rot90_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.rot90' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.rot90'.")

check_valid('tf.experimental.numpy.rot90', generated_inputs['tf.experimental.numpy.rot90'], lib="tf", suffix=0)
