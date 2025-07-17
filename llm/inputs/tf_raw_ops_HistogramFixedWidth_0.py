
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_histogram_fixed_width_inputs():
    list_of_inputs = []

    # Input 1: Basic case with integers
    values = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    value_range = np.array([0, 6], dtype=np.int32)
    nbins = np.array(6, dtype=np.int32)
    dtype = tf.int32
    name = "histogram_int"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Floats with different range
    values = np.array([1.5, 2.7, 3.1, 4.9, 5.2], dtype=np.float32)
    value_range = np.array([1.0, 6.0], dtype=np.float32)
    nbins = np.array(5, dtype=np.int32)
    dtype = tf.int32
    name = "histogram_float"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values and wider range
    values = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    value_range = np.array([-3, 3], dtype=np.int64)
    nbins = np.array(6, dtype=np.int32)
    dtype = tf.int32
    name = "histogram_negative"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: nbins = 1
    values = np.array([1, 2, 3], dtype=np.int32)
    value_range = np.array([0, 4], dtype=np.int32)
    nbins = np.array(1, dtype=np.int32)
    dtype = tf.int32
    name = "histogram_one_bin"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger nbins
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    value_range = np.array([0.0, 1.0], dtype=np.float32)
    nbins = np.array(10, dtype=np.int32)
    dtype = tf.int32
    name = "histogram_large_nbins"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different dtype for output
    values = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    value_range = np.array([0, 6], dtype=np.int32)
    nbins = np.array(6, dtype=np.int32)
    dtype = tf.int64
    name = "histogram_int64"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty values array
    values = np.array([], dtype=np.int32)
    value_range = np.array([0, 6], dtype=np.int32)
    nbins = np.array(6, dtype=np.int32)
    dtype = tf.int32
    name = "histogram_empty"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Values outside the range
    values = np.array([-5, 0, 5, 10], dtype=np.int32)
    value_range = np.array([0, 5], dtype=np.int32)
    nbins = np.array(5, dtype=np.int32)
    dtype = tf.int32
    name = "histogram_out_of_range"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large values and nbins
    values = np.array([1000, 2000, 3000], dtype=np.int32)
    value_range = np.array([0, 4000], dtype=np.int32)
    nbins = np.array(4, dtype=np.int32)
    dtype = tf.int32
    name = "histogram_large_values"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Value range with floats and large values
    values = np.array([100.5, 200.7, 300.1], dtype=np.float32)
    value_range = np.array([50.0, 350.0], dtype=np.float32)
    nbins = np.array(10, dtype=np.int32)
    dtype = tf.int32
    name = "histogram_float_large_range"
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
