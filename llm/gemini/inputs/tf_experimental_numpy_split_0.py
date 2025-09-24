
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_split_inputs():
    list_of_inputs = []

    # Input 1
    ary = tf.constant(np.arange(8).reshape((2, 4))).numpy()
    indices_or_sections = 2
    axis = 1
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ary = tf.constant(np.arange(9).reshape((3, 3))).numpy()
    indices_or_sections = 3
    axis = 0
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    ary = tf.constant(np.arange(16).reshape((4, 4))).numpy()
    indices_or_sections = 4
    axis = 1
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    ary = tf.constant(np.arange(24).reshape((2, 3, 4))).numpy()
    indices_or_sections = 2
    axis = 0
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    ary = tf.constant(np.arange(24).reshape((2, 3, 4))).numpy()
    indices_or_sections = 3
    axis = 1
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    ary = tf.constant(np.arange(24).reshape((2, 3, 4))).numpy()
    indices_or_sections = 4
    axis = 2
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    ary = tf.constant(np.arange(10)).numpy()
    indices_or_sections = 5
    axis = 0
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    ary = tf.constant(np.arange(12).reshape((2,2,3))).numpy()
    indices_or_sections = 2
    axis = 1
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    ary = tf.constant(np.arange(12).reshape((2,2,3))).numpy()
    indices_or_sections = 3
    axis = 2
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    ary = tf.constant(np.arange(48).reshape((2,3,2,4))).numpy()
    indices_or_sections = 2
    axis = 3
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.split"] = tf_experimental_numpy_split_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.split' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.split'.")

check_valid('tf.experimental.numpy.split', generated_inputs['tf.experimental.numpy.split'], lib="tf", suffix=0)
