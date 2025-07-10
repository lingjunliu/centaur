
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_histogram_fixed_width_bins_inputs():
    list_of_inputs = []

    # Input 1
    values = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)
    value_range = tf.constant([0, 6], dtype=tf.int32)
    nbins = 6
    dtype = tf.int32
    name = "histogram1"
    input_dict = {"values": values.numpy(), "value_range": value_range.numpy(), "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    values = tf.constant([-1, 0, 1, 2, 3], dtype=tf.int32)
    value_range = tf.constant([-2, 4], dtype=tf.int32)
    nbins = 6
    dtype = tf.int32
    name = "histogram2"
    input_dict = {"values": values.numpy(), "value_range": value_range.numpy(), "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    values = tf.constant([1, 2, 3, 4], dtype=tf.int32)
    value_range = tf.constant([1, 5], dtype=tf.int32)
    nbins = 4
    dtype = tf.int32
    name = "histogram3"
    input_dict = {"values": values.numpy(), "value_range": value_range.numpy(), "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    values = tf.constant([0, 0, 0, 0, 0], dtype=tf.int32)
    value_range = tf.constant([0, 1], dtype=tf.int32)
    nbins = 5
    dtype = tf.int32
    name = "histogram4"
    input_dict = {"values": values.numpy(), "value_range": value_range.numpy(), "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    values = tf.constant([10, 20, 30, 40, 50], dtype=tf.int32)
    value_range = tf.constant([0, 60], dtype=tf.int32)
    nbins = 6
    dtype = tf.int32
    name = "histogram5"
    input_dict = {"values": values.numpy(), "value_range": value_range.numpy(), "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    values = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)
    value_range = tf.constant([0, 5], dtype=tf.int32)
    nbins = 5
    dtype = tf.int64
    name = "histogram6"
    input_dict = {"values": values.numpy(), "value_range": value_range.numpy(), "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    values = tf.constant([[-1, 0], [1, 2]], dtype=tf.int32)
    value_range = tf.constant([-2, 3], dtype=tf.int32)
    nbins = 5
    dtype = tf.int32
    name = "histogram7"
    input_dict = {"values": values.numpy(), "value_range": value_range.numpy(), "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    values = tf.constant([[-1, -2], [3, 4]], dtype=tf.int32)
    value_range = tf.constant([-3, 5], dtype=tf.int32)
    nbins = 8
    dtype = tf.int32
    name = "histogram8"
    input_dict = {"values": values.numpy(), "value_range": value_range.numpy(), "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    values = tf.constant([1, 1, 1, 1, 1], dtype=tf.int32)
    value_range = tf.constant([1, 2], dtype=tf.int32)
    nbins = 5
    dtype = tf.int32
    name = "histogram9"
    input_dict = {"values": values.numpy(), "value_range": value_range.numpy(), "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    values = tf.constant([1, 1, 1, 1, 1], dtype=tf.int32)
    value_range = tf.constant([1, 2], dtype=tf.int32)
    nbins = 1
    dtype = tf.int32
    name = "histogram10"
    input_dict = {"values": values.numpy(), "value_range": value_range.numpy(), "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.histogram_fixed_width_bins"] = tf_histogram_fixed_width_bins_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.histogram_fixed_width_bins' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.histogram_fixed_width_bins'.")

check_valid('tf.histogram_fixed_width_bins', generated_inputs['tf.histogram_fixed_width_bins'], lib="tf", suffix=0)
