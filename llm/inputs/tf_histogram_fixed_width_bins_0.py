
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_histogram_fixed_width_bins_inputs():
    list_of_inputs = []

    # Input 1: Basic test case
    values = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    value_range = np.array([0.0, 5.0], dtype=np.float32)
    nbins = 5
    dtype = tf.int32
    name = "basic_histogram"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative values
    values = np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    value_range = np.array([-2.0, 4.0], dtype=np.float32)
    nbins = 6
    dtype = tf.int32
    name = "negative_values"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different nbins
    values = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    value_range = np.array([0, 6], dtype=np.float32)
    nbins = 3
    dtype = tf.int32
    name = "different_nbins"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Values outside range
    values = np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6], dtype=np.float32)
    value_range = np.array([0, 4], dtype=np.float32)
    nbins = 4
    dtype = tf.int32
    name = "values_outside_range"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small value range
    values = np.array([1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float32)
    value_range = np.array([1.0, 2.0], dtype=np.float32)
    nbins = 5
    dtype = tf.int32
    name = "small_value_range"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All values the same
    values = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    value_range = np.array([0.0, 5.0], dtype=np.float32)
    nbins = 5
    dtype = tf.int32
    name = "all_same_values"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Edge case with value_range
    values = np.array([0.0, 5.0], dtype=np.float32)
    value_range = np.array([0.0, 5.0], dtype=np.float32)
    nbins = 5
    dtype = tf.int32
    name = "edge_value_range"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D values
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    value_range = np.array([0.0, 5.0], dtype=np.float32)
    nbins = 5
    dtype = tf.int32
    name = "2d_values"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger number of bins
    values = np.array([i for i in range(100)], dtype=np.float32)
    value_range = np.array([0.0, 100.0], dtype=np.float32)
    nbins = 200
    dtype = tf.int32
    name = "larger_nbins"
    input_dict = {"values": values, "value_range": value_range, "nbins": nbins, "dtype": dtype, "name": name}
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
