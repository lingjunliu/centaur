
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.experimental.numpy.experimental_enable_numpy_behavior()

def tf_np_logical_and_inputs():
    list_of_inputs = []

    # Input 1: Basic boolean tensors
    x1 = tf.constant([True, False, True, False]).numpy()
    x2 = tf.constant([True, True, False, False]).numpy()
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensors (treated as booleans)
    x1 = tf.constant([1, 0, 1, 0]).numpy()
    x2 = tf.constant([1, 1, 0, 0]).numpy()
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float tensors (treated as booleans)
    x1 = tf.constant([1.0, 0.0, 1.5, -0.5]).numpy()
    x2 = tf.constant([0.5, 1.0, 0.0, 0.0]).numpy()
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D boolean tensors
    x1 = tf.constant([[True, False], [True, True]]).numpy()
    x2 = tf.constant([[True, True], [False, True]]).numpy()
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D integer tensors
    x1 = tf.constant([[1, 0], [1, 1]]).numpy()
    x2 = tf.constant([[1, 1], [0, 1]]).numpy()
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shapes (broadcasting)
    x1 = tf.constant([True, False]).numpy()
    x2 = tf.constant([[True, True], [False, False]]).numpy()
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different shapes (broadcasting) - integers
    x1 = tf.constant([1, 0]).numpy()
    x2 = tf.constant([[1, 1], [0, 0]]).numpy()
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D tensors
    x1 = tf.constant([[[True, False], [True, True]], [[False, True], [True, False]]]).numpy()
    x2 = tf.constant([[[True, True], [False, True]], [[True, False], [False, False]]]).numpy()
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All False values
    x1 = tf.constant([False, False, False]).numpy()
    x2 = tf.constant([False, False, False]).numpy()
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.logical_and"] = tf_np_logical_and_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.logical_and' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.logical_and'.")

check_valid('tf.experimental.numpy.logical_and', generated_inputs['tf.experimental.numpy.logical_and'], lib="tf", suffix=0)
