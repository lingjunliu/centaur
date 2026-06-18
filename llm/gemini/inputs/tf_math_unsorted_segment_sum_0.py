
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_unsorted_segment_sum_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 array
    input_dict = {
        "data": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "segment_ids": np.array([0, 1, 0, 2, 1], dtype=np.int32),
        "num_segments": 3,
        "name": "basic_1d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array with standard segments
    input_dict = {
        "data": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        "segment_ids": np.array([0, 1, 0], dtype=np.int32),
        "num_segments": 2,
        "name": "basic_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Using negative segment IDs (values are ignored)
    input_dict = {
        "data": np.array([10, 20, 30, 40], dtype=np.int64),
        "segment_ids": np.array([-1, 0, -1, 1], dtype=np.int64),
        "num_segments": 2,
        "name": "negative_indices"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64 data with 1D segment IDs
    input_dict = {
        "data": np.ones((2, 3, 2), dtype=np.float64),
        "segment_ids": np.array([0, 1], dtype=np.int32),
        "num_segments": 3,
        "name": "3d_data"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float64 data, int64 segment IDs, and unused segments
    input_dict = {
        "data": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        "segment_ids": np.array([1, 1, 0], dtype=np.int64),
        "num_segments": 4,
        "name": "unused_segments"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single-element float32 array
    input_dict = {
        "data": np.array([1.5], dtype=np.float32),
        "segment_ids": np.array([0], dtype=np.int32),
        "num_segments": 1,
        "name": "single_element"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Out-of-order segment IDs
    input_dict = {
        "data": np.array([1, 2, 3, 4, 5], dtype=np.int16),
        "segment_ids": np.array([4, 2, 0, 1, 3], dtype=np.int32),
        "num_segments": 5,
        "name": "out_of_order"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex64 data types
    input_dict = {
        "data": np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64),
        "segment_ids": np.array([0, 1, 0], dtype=np.int32),
        "num_segments": 2,
        "name": "complex_data"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multi-dimensional segment IDs (prefix of shape)
    input_dict = {
        "data": np.ones((2, 2, 3), dtype=np.float32),
        "segment_ids": np.array([[0, 1], [1, 0]], dtype=np.int32),
        "num_segments": 2,
        "name": "multi_dim_ids"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Sparse segment IDs with a large num_segments
    input_dict = {
        "data": np.array([5, 10], dtype=np.int32),
        "segment_ids": np.array([1, 3], dtype=np.int32),
        "num_segments": 5,
        "name": "sparse_segments"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.unsorted_segment_sum"] = tf_math_unsorted_segment_sum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.unsorted_segment_sum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.unsorted_segment_sum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.unsorted_segment_sum', generated_inputs['tf.math.unsorted_segment_sum'], lib="tf", suffix=0)
