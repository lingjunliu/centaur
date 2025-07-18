
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_ix__inputs():
    list_of_inputs = []

    # The error "ValueError: Arguments must be 1-d, got arg 0 of rank 2"
    # indicates that the function is being called with a single 2D tensor,
    # whereas it expects one or more 1D tensors. The test harness appears
    # to convert the entire 'args' value into a single tensor and pass that
    # as the only argument. To make this work, we must provide a single 1D array
    # as the value for 'args'. The function tf.experimental.numpy.ix_ is valid
    # when called with a single 1D array.

    # Input 1: A simple 1D integer array
    input_dict = {'args': np.array([1, 4, 2], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: An empty 1D array
    input_dict = {'args': np.array([], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: A 1D boolean array
    input_dict = {'args': np.array([True, False, True, False])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: A 1D array with a single element
    input_dict = {'args': np.array([10], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: A 1D array with negative values and zero
    input_dict = {'args': np.array([0, -1, 5, -10, 2], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: A 1D array with duplicate values, unsorted
    input_dict = {'args': np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: A longer 1D integer array
    input_dict = {'args': np.arange(10, dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: A 1D array of int8
    input_dict = {'args': np.array([1, 2, 3, 4, 5], dtype=np.int8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: A 1D array of int16
    input_dict = {'args': np.array([100, 200, 300], dtype=np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: A 1D array with large integer values
    input_dict = {'args': np.array([100000, 200000, 300000], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.ix_"] = tf_experimental_numpy_ix__inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.ix_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.ix_'.")

check_valid('tf.experimental.numpy.ix_', generated_inputs['tf.experimental.numpy.ix_'], lib="tf", suffix=0)
