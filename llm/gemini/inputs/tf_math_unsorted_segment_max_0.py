
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()

def tf_math_unsorted_segment_max_inputs():
    list_of_inputs = []

    # Case 1: Simple 1D float32
    list_of_inputs.append({
        "data": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "segment_ids": np.array([0, 0, 1, 1], dtype=np.int32),
        "num_segments": 2,
        "name": "segment_max_1d_float"
    })

    # Case 2: 2D data (3x4), 1D segment_ids (3)
    list_of_inputs.append({
        "data": np.array([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]], dtype=np.int32),
        "segment_ids": np.array([0, 1, 0], dtype=np.int32),
        "num_segments": 2,
        "name": "segment_max_2d_int"
    })

    # Case 3: 3D data (2x2x3), 1D segment_ids (2)
    list_of_inputs.append({
        "data": np.array([[[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], [[7.5, 8.5, 9.5], [10.5, 11.5, 12.5]]], dtype=np.float64),
        "segment_ids": np.array([1, 0], dtype=np.int32),
        "num_segments": 2,
        "name": "segment_max_3d_float"
    })

    # Case 4: 3D data (2x2x2), 2D segment_ids (2x2)
    list_of_inputs.append({
        "data": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
        "segment_ids": np.array([[0, 1], [0, 1]], dtype=np.int32),
        "num_segments": 2,
        "name": "segment_max_3d_2d_ids"
    })

    # Case 5: 2D data with negative values and some segment_ids negative (should be ignored)
    list_of_inputs.append({
        "data": np.array([[-1.0, -2.0], [3.0, 4.0], [-5.0, -6.0]], dtype=np.float32),
        "segment_ids": np.array([-1, 0, -1], dtype=np.int32),
        "num_segments": 1,
        "name": "segment_max_neg_ids"
    })

    # Case 6: 1D data with int64 and int64 segment_ids
    list_of_inputs.append({
        "data": np.array([10, 20, 30, 40, 50], dtype=np.int64),
        "segment_ids": np.array([0, 1, 2, 1, 0], dtype=np.int64),
        "num_segments": 3,
        "name": "segment_max_int64"
    })

    # Case 7: 2D data with uint8
    list_of_inputs.append({
        "data": np.array([[10, 20], [30, 40], [50, 60]], dtype=np.uint8),
        "segment_ids": np.array([0, 0, 1], dtype=np.int32),
        "num_segments": 2,
        "name": "segment_max_uint8"
    })

    # Case 8: 4D data (2x2x2x2), 2D segment_ids (2x2)
    list_of_inputs.append({
        "data": np.ones((2, 2, 2, 2), dtype=np.float32),
        "segment_ids": np.array([[0, 1], [1, 0]], dtype=np.int32),
        "num_segments": 3,
        "name": "segment_max_4d"
    })

    # Case 9: 1D data with empty segment (num_segments larger than max segment_id)
    list_of_inputs.append({
        "data": np.array([1.5, 2.5], dtype=np.float32),
        "segment_ids": np.array([0, 0], dtype=np.int32),
        "num_segments": 3,
        "name": "segment_max_empty_seg"
    })

    # Case 10: 3D data, 3D segment_ids (fully specified segments)
    list_of_inputs.append({
        "data": np.array([[[1], [2]], [[3], [4]]], dtype=np.float32),
        "segment_ids": np.array([[[1], [0]], [[0], [1]]], dtype=np.int32),
        "num_segments": 2,
        "name": "segment_max_full_ids"
    })

    return list_of_inputs

generated_inputs["tf.math.unsorted_segment_max"] = tf_math_unsorted_segment_max_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.unsorted_segment_max' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.unsorted_segment_max'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.unsorted_segment_max', generated_inputs['tf.math.unsorted_segment_max'], lib="tf", suffix=0)
