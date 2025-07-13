
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_histogram_fixed_width_inputs():
    list_of_inputs = []

    # Input 1: Basic example with integers
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    value_range = np.array([0, 10], dtype=np.int32)
    nbins = np.array(5, dtype=np.int32)
    dtype = tf.int32
    name = "histogram_1"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float values
    values = np.array([1.0, 2.5, 3.7, 4.2, 5.9], dtype=np.float32)
    value_range = np.array([0.0, 6.0], dtype=np.float32)
    nbins = np.array(6, dtype=np.int32)
    dtype = tf.int32
    name = "histogram_2"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values
    values = np.array([-5, -3, -1, 1, 3, 5], dtype=np.int32)
    value_range = np.array([-6, 6], dtype=np.int32)
    nbins = np.array(12, dtype=np.int32)
    dtype = tf.int32
    name = "histogram_3"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different dtype for histogram
    values = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    value_range = np.array([0, 6], dtype=np.int32)
    nbins = np.array(3, dtype=np.int32)
    dtype = tf.int64
    name = "histogram_4"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger number of bins
    values = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    value_range = np.array([0, 100], dtype=np.int32)
    nbins = np.array(50, dtype=np.int32)
    dtype = tf.int32
    name = "histogram_5"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: value_range is same
    values = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    value_range = np.array([5, 5], dtype=np.int32)
    nbins = np.array(5, dtype=np.int32)
    dtype = tf.int32
    name = "histogram_6"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64 values
    values = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    value_range = np.array([0, 6], dtype=np.int64)
    nbins = np.array(3, dtype=np.int32)
    dtype = tf.int32
    name = "histogram_7"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64 values
    values = np.array([1.0, 2.5, 3.7, 4.2, 5.9], dtype=np.float64)
    value_range = np.array([0.0, 6.0], dtype=np.float64)
    nbins = np.array(6, dtype=np.int32)
    dtype = tf.int32
    name = "histogram_8"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: Large values and range
    values = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int32)
    value_range = np.array([0, 6000], dtype=np.int32)
    nbins = np.array(6, dtype=np.int32)
    dtype = tf.int32
    name = "histogram_9"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty values array
    values = np.array([], dtype=np.int32)
    value_range = np.array([0, 10], dtype=np.int32)
    nbins = np.array(5, dtype=np.int32)
    dtype = tf.int32
    name = "histogram_10"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.HistogramFixedWidth"] = tf_raw_ops_histogram_fixed_width_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.HistogramFixedWidth' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.HistogramFixedWidth'.")

check_valid('tf.raw_ops.HistogramFixedWidth', generated_inputs['tf.raw_ops.HistogramFixedWidth'], lib="tf", suffix=0)
