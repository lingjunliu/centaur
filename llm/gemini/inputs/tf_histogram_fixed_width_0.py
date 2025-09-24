
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_histogram_fixed_width_inputs():
    list_of_inputs = []

    # Input 1
    values = tf.constant(np.array([-1.0, 0.0, 1.5, 2.0, 5.0, 15], dtype=np.float32))
    value_range = tf.constant(np.array([0.0, 5.0], dtype=np.float32))
    nbins = np.int32(5)
    dtype = tf.int32
    name = "histogram_1"

    input_dict = {
        "values": values.numpy(),
        "value_range": value_range.numpy(),
        "nbins": nbins,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    values = tf.constant(np.array([1, 2, 3, 4, 5], dtype=np.int32))
    value_range = tf.constant(np.array([0, 6], dtype=np.int32))
    nbins = np.int32(6)
    dtype = tf.int32
    name = "histogram_2"

    input_dict = {
        "values": values.numpy(),
        "value_range": value_range.numpy(),
        "nbins": nbins,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    values = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    value_range = tf.constant(np.array([0, 5], dtype=np.int32))
    nbins = np.int32(5)
    dtype = tf.int32
    name = "histogram_3"

    input_dict = {
        "values": values.numpy(),
        "value_range": value_range.numpy(),
        "nbins": nbins,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    values = tf.constant(np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32))
    value_range = tf.constant(np.array([0.0, 1.0], dtype=np.float32))
    nbins = np.int32(10)
    dtype = tf.int32
    name = "histogram_4"

    input_dict = {
        "values": values.numpy(),
        "value_range": value_range.numpy(),
        "nbins": nbins,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    values = tf.constant(np.array([-1, -2, -3, -4, -5], dtype=np.int32))
    value_range = tf.constant(np.array([-6, 0], dtype=np.int32))
    nbins = np.int32(6)
    dtype = tf.int32
    name = "histogram_5"

    input_dict = {
        "values": values.numpy(),
        "value_range": value_range.numpy(),
        "nbins": nbins,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    values = tf.constant(np.array([1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0], dtype=np.float32))
    value_range = tf.constant(np.array([0.0, 5.0], dtype=np.float32))
    nbins = np.int32(5)
    dtype = tf.int32
    name = "histogram_6"

    input_dict = {
        "values": values.numpy(),
        "value_range": value_range.numpy(),
        "nbins": nbins,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    values = tf.constant(np.array([0], dtype=np.int32))
    value_range = tf.constant(np.array([0, 1], dtype=np.int32))
    nbins = np.int32(1)
    dtype = tf.int32
    name = "histogram_7"

    input_dict = {
        "values": values.numpy(),
        "value_range": value_range.numpy(),
        "nbins": nbins,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    values = tf.constant(np.array([100, 200, 300], dtype=np.int32))
    value_range = tf.constant(np.array([0, 400], dtype=np.int32))
    nbins = np.int32(4)
    dtype = tf.int32
    name = "histogram_8"

    input_dict = {
        "values": values.numpy(),
        "value_range": value_range.numpy(),
        "nbins": nbins,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    values = tf.constant(np.array([], dtype=np.int32))
    value_range = tf.constant(np.array([0, 10], dtype=np.int32))
    nbins = np.int32(10)
    dtype = tf.int32
    name = "histogram_9"

    input_dict = {
        "values": values.numpy(),
        "value_range": value_range.numpy(),
        "nbins": nbins,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    values = tf.constant(np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32))
    value_range = tf.constant(np.array([1.0, 5.0], dtype=np.float32))
    nbins = np.int32(4)
    dtype = tf.int32
    name = "histogram_10"

    input_dict = {
        "values": values.numpy(),
        "value_range": value_range.numpy(),
        "nbins": nbins,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.histogram_fixed_width"] = tf_histogram_fixed_width_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.histogram_fixed_width' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.histogram_fixed_width'.")

check_valid('tf.histogram_fixed_width', generated_inputs['tf.histogram_fixed_width'], lib="tf", suffix=0)
