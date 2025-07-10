
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_append_inputs():
    list_of_inputs = []

    # Input 1
    arr = tf.constant([1, 2, 3])
    values = tf.constant([4, 5, 6])
    axis = None
    input_dict = {"arr": arr.numpy(), "values": values.numpy(), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    arr = tf.constant([[1, 2], [3, 4]])
    values = tf.constant([[5, 6], [7,8]])
    axis = 0
    input_dict = {"arr": arr.numpy(), "values": values.numpy(), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    arr = tf.constant([[1, 2], [3, 4]])
    values = tf.constant([[5], [6]])
    axis = 1
    input_dict = {"arr": arr.numpy(), "values": values.numpy(), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    arr = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    values = tf.constant([[[9, 10], [11, 12]], [[13,14],[15,16]]])
    axis = 0
    input_dict = {"arr": arr.numpy(), "values": values.numpy(), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    arr = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    values = tf.constant([[[9, 10], [11, 12]],[[13, 14], [15, 16]]])
    axis = 1
    input_dict = {"arr": arr.numpy(), "values": values.numpy(), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    arr = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    values = tf.constant([[[9], [10]],[[11],[12]]])
    axis = 2
    input_dict = {"arr": arr.numpy(), "values": values.numpy(), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    arr = tf.constant([1, 2, 3], dtype=tf.float32)
    values = tf.constant([4, 5, 6], dtype=tf.float32)
    axis = None
    input_dict = {"arr": arr.numpy(), "values": values.numpy(), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    arr = tf.constant([[1, 2], [3, 4]], dtype=tf.int64)
    values = tf.constant([[5, 6], [7, 8]], dtype=tf.int64)
    axis = 0
    input_dict = {"arr": arr.numpy(), "values": values.numpy(), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    arr = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.complex64)
    values = tf.constant([[[9], [10]], [[11], [12]]], dtype=tf.complex64)
    axis = 2
    input_dict = {"arr": arr.numpy(), "values": values.numpy(), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    arr = tf.constant([1, 2, 3, 4])
    values = tf.constant([5,6,7,8])
    axis = None
    input_dict = {"arr": arr.numpy(), "values": values.numpy(), "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.append"] = tf_experimental_numpy_append_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.append' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.append'.")

check_valid('tf.experimental.numpy.append', generated_inputs['tf.experimental.numpy.append'], lib="tf", suffix=0)
