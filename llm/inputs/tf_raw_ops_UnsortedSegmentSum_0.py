
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_unsorted_segment_sum_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array([1, 2, 3, 4]).astype(np.float32)
    segment_ids = np.array([0, 0, 1, 1]).astype(np.int32)
    num_segments = np.array(2).astype(np.int32)
    name = "test_sum_1"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]]).astype(np.int32)
    segment_ids = np.array([0, 1, 0]).astype(np.int32)
    num_segments = np.array(2).astype(np.int32)
    name = "test_sum_2"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array([1, 2, 3, 4]).astype(np.float64)
    segment_ids = np.array([0, 0, 1, 0]).astype(np.int64)
    num_segments = np.array(2).astype(np.int32)
    name = "test_sum_3"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array([1, 2, 3, 4, 5, 6]).astype(np.int64)
    segment_ids = np.array([0, 1, 2, 0, 1, 0]).astype(np.int32)
    num_segments = np.array(3).astype(np.int32)
    name = "test_sum_4"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different data type
    data = np.array([1, 2, 3, 4]).astype(np.uint8)
    segment_ids = np.array([0, 0, 1, 1]).astype(np.int32)
    num_segments = np.array(2).astype(np.int32)
    name = "test_sum_5"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: More dimensions
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.int32)
    segment_ids = np.array([0, 1]).astype(np.int32)
    num_segments = np.array(2).astype(np.int32)
    name = "test_sum_6"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: num_segments > max(segment_ids)
    data = np.array([1, 2, 3]).astype(np.int32)
    segment_ids = np.array([0, 1, 0]).astype(np.int32)
    num_segments = np.array(5).astype(np.int32)
    name = "test_sum_7"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64 data and segment_ids
    data = np.array([1, 2, 3, 4]).astype(np.int64)
    segment_ids = np.array([0, 0, 1, 1]).astype(np.int64)
    num_segments = np.array(2).astype(np.int32)
    name = "test_sum_8"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint16 data
    data = np.array([1, 2, 3, 4]).astype(np.uint16)
    segment_ids = np.array([0, 0, 1, 1]).astype(np.int32)
    num_segments = np.array(2).astype(np.int32)
    name = "test_sum_9"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: bfloat16 data
    data = np.array([1, 2, 3, 4]).astype(np.float16)
    segment_ids = np.array([0, 0, 1, 1]).astype(np.int32)
    num_segments = np.array(2).astype(np.int32)
    name = "test_sum_11"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Empty data and segment_ids with num_segments=1
    data = np.array([]).astype(np.float32)
    segment_ids = np.array([]).astype(np.int32)
    num_segments = np.array(1).astype(np.int32)
    name = "test_sum_12"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.UnsortedSegmentSum"] = tf_raw_ops_unsorted_segment_sum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.UnsortedSegmentSum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.UnsortedSegmentSum'.")

check_valid('tf.raw_ops.UnsortedSegmentSum', generated_inputs['tf.raw_ops.UnsortedSegmentSum'], lib="tf", suffix=0)
