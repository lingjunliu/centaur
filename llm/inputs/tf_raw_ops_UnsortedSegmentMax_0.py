
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_unsortedsegmentmax_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]], dtype=np.int32)
    segment_ids = np.array([0, 1, 0], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    segment_ids = np.array([0, 1, 0], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = np.array([1, 2, 3, 4], dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    num_segments = np.array(2, dtype=np.int64)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    segment_ids = np.array([0, 1, 0], dtype=np.int64)
    num_segments = np.array(2, dtype=np.int64)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different num_segments
    data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    num_segments = np.array(3, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty segment
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 2, 2], dtype=np.int32)
    num_segments = np.array(3, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Higher dimension data
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint8
    data = np.array([1, 2, 3, 4], dtype=np.uint8)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.UnsortedSegmentMax"] = tf_raw_ops_unsortedsegmentmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.UnsortedSegmentMax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.UnsortedSegmentMax'.")

check_valid('tf.raw_ops.UnsortedSegmentMax', generated_inputs['tf.raw_ops.UnsortedSegmentMax'], lib="tf", suffix=0)
