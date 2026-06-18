
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_histogram_fixed_width_bins_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 input with 5 bins
    input_dict = {
        'values': np.array([-1.0, 0.0, 1.5, 2.0, 5.0, 15.0], dtype=np.float32),
        'value_range': np.array([0.0, 5.0], dtype=np.float32),
        'nbins': 5,
        'dtype': np.int32,
        'name': 'bins_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 input, 10 bins, int64 output dtype
    input_dict = {
        'values': np.array([[-10.0, -5.0], [0.0, 5.0]], dtype=np.float64),
        'value_range': np.array([-10.0, 10.0], dtype=np.float64),
        'nbins': 10,
        'dtype': np.int64,
        'name': 'bins_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float32 values, 5 bins, negative and positive
    input_dict = {
        'values': np.array([[[1.0, 2.0], [3.0, -4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32),
        'value_range': np.array([-5.0, 10.0], dtype=np.float32),
        'nbins': 5,
        'dtype': np.int32,
        'name': 'bins_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Single element float32 array, many bins
    input_dict = {
        'values': np.array([0.55], dtype=np.float32),
        'value_range': np.array([0.0, 1.0], dtype=np.float32),
        'nbins': 100,
        'dtype': np.int32,
        'name': 'bins_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float32 values with negative range bounds
    input_dict = {
        'values': np.array([-3.5, -2.1, -1.0, -0.5], dtype=np.float32),
        'value_range': np.array([-5.0, -1.0], dtype=np.float32),
        'nbins': 4,
        'dtype': np.int32,
        'name': 'bins_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Scalar/0D float32 value
    input_dict = {
        'values': np.array(2.5, dtype=np.float32),
        'value_range': np.array([0.0, 5.0], dtype=np.float32),
        'nbins': 5,
        'dtype': np.int32,
        'name': 'bins_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 values with very large range and bin size
    input_dict = {
        'values': np.array([1e5, 2e5, 3e5], dtype=np.float64),
        'value_range': np.array([0.0, 1e6], dtype=np.float64),
        'nbins': 1000,
        'dtype': np.int32,
        'name': 'bins_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float64 values, different bounds
    input_dict = {
        'values': np.array([-1.5, 0.5, 1.5], dtype=np.float64),
        'value_range': np.array([-2.0, 2.0], dtype=np.float64),
        'nbins': 4,
        'dtype': np.int32,
        'name': 'bins_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High-dimensional float32 array using generated random values
    input_dict = {
        'values': np.random.uniform(-10, 10, (2, 2, 2, 2)).astype(np.float32),
        'value_range': np.array([-10.0, 10.0], dtype=np.float32),
        'nbins': 10,
        'dtype': np.int32,
        'name': 'bins_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float64 values and Int64 output dtype
    input_dict = {
        'values': np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        'value_range': np.array([0.0, 100.0], dtype=np.float64),
        'nbins': 10,
        'dtype': np.int64,
        'name': 'bins_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.histogram_fixed_width_bins"] = tf_histogram_fixed_width_bins_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.histogram_fixed_width_bins' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.histogram_fixed_width_bins'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.histogram_fixed_width_bins', generated_inputs['tf.histogram_fixed_width_bins'], lib="tf", suffix=0)
