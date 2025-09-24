
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_unsorted_segment_min_inputs():
    list_of_inputs = []

    # Input 1: Basic example with int32 data and segment_ids
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"name": "test1", "data": data, "segment_ids": segment_ids, "num_segments": num_segments}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float32 data with int64 segment_ids
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"name": "test2", "data": data, "segment_ids": segment_ids, "num_segments": num_segments}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional data
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"name": "test3", "data": data, "segment_ids": segment_ids, "num_segments": num_segments}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Single segment
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 0], dtype=np.int32)
    num_segments = np.array(1, dtype=np.int32)
    input_dict = {"name": "test4", "data": data, "segment_ids": segment_ids, "num_segments": num_segments}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: All different segments
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 1, 2, 3], dtype=np.int32)
    num_segments = np.array(4, dtype=np.int32)
    input_dict = {"name": "test5", "data": data, "segment_ids": segment_ids, "num_segments": num_segments}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6:  uint8 data
    data = np.array([1, 2, 3, 4], dtype=np.uint8)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"name": "test6", "data": data, "segment_ids": segment_ids, "num_segments": num_segments}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty segment
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(3, dtype=np.int32)
    input_dict = {"name": "test7", "data": data, "segment_ids": segment_ids, "num_segments": num_segments}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger number of segments than data points
    data = np.array([1, 2], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    num_segments = np.array(5, dtype=np.int32)
    input_dict = {"name": "test8", "data": data, "segment_ids": segment_ids, "num_segments": num_segments}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int64 data
    data = np.array([1, 2, 3, 4], dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"name": "test9", "data": data, "segment_ids": segment_ids, "num_segments": num_segments}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.UnsortedSegmentMin"] = tf_raw_ops_unsorted_segment_min_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.UnsortedSegmentMin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.UnsortedSegmentMin'.")

check_valid('tf.raw_ops.UnsortedSegmentMin', generated_inputs['tf.raw_ops.UnsortedSegmentMin'], lib="tf", suffix=0)
