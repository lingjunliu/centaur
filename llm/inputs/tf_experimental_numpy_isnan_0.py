
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_isnan_inputs():
    list_of_inputs = []

    # Input 1: Basic float array with NaN
    x = np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Array with only NaNs
    x = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Array with no NaNs
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional array with mixed values
    x = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Array with positive and negative NaNs
    x = np.array([np.nan, -np.nan, 1.0], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array with NaNs
    x = np.array([[[1.0, np.nan], [3.0, 4.0]], [[5.0, 6.0], [np.nan, 8.0]]], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero-dimensional array (scalar) with NaN
    x = np.array(np.nan, dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array with a mix of positive, negative, zero, and NaN
    x = np.array([1.0, -1.0, 0.0, np.nan, -np.nan], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger array
    x = np.random.randn(10, 10)
    x[0,0] = np.nan
    x[5,5] = np.nan
    x[9,9] = np.nan
    x = x.astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Higher dimensional array
    x = np.random.randn(2, 3, 4, 5)
    x[0,0,0,0] = np.nan
    x[1,2,3,4] = np.nan
    x = x.astype(np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.isnan"] = tf_experimental_numpy_isnan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.isnan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.isnan'.")

check_valid('tf.experimental.numpy.isnan', generated_inputs['tf.experimental.numpy.isnan'], lib="tf", suffix=0)
