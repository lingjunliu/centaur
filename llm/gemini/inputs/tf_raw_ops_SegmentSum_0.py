
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_segment_sum_inputs():
    list_of_inputs = []

    # Input 1: Basic example with int32 data and segment_ids
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "basic_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 data and segment_ids
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    segment_ids = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "basic_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int64 segment_ids
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "int64_segment_ids"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data type (float64)
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "float64_data"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: One segment ID
    data = np.array([1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 0], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "one_segment"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: No segments
    data = np.array([1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 1, 2], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "no_segments"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multi-dimensional data
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "multi_dim_data"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger segment IDs
    data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    segment_ids = np.array([1, 1, 2, 3, 3], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "larger_segment_ids"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint8 data type
    data = np.array([1, 2, 3, 4], dtype=np.uint8)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "uint8_data"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: More complex multi-dimensional data
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.int32)
    segment_ids = np.array([0, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "complex_multi_dim"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: bfloat16 data type
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "bfloat16_data"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SegmentSum"] = tf_raw_ops_segment_sum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SegmentSum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SegmentSum'.")

check_valid('tf.raw_ops.SegmentSum', generated_inputs['tf.raw_ops.SegmentSum'], lib="tf", suffix=0)
