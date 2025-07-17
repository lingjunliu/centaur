
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_UnsortedSegmentSum_inputs():
    list_of_inputs = []

    # Input 1: Simple case with positive integers
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "input1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data type (float32)
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "input2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Unsorted segment_ids
    data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    segment_ids = np.array([1, 0, 2, 1, 0], dtype=np.int32)
    num_segments = np.array(3, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "input3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative segment_ids (should be ignored)
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([-1, 0, -1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "input4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional data
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "input5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different segment_ids type (int64)
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "input6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different num_segments type (int64)
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int64)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "input7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty segment for a segment_id
    data = np.array([1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 2, 0], dtype=np.int32)
    num_segments = np.array(3, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "input8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Data with a different dtype (uint8)
    data = np.array([1, 2, 3, 4], dtype=np.uint8)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "input9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D data
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.int32)
    segment_ids = np.array([0, 1, 0], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "input10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.UnsortedSegmentSum"] = tf_raw_ops_UnsortedSegmentSum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.UnsortedSegmentSum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.UnsortedSegmentSum'.")

check_valid('tf.raw_ops.UnsortedSegmentSum', generated_inputs['tf.raw_ops.UnsortedSegmentSum'], lib="tf", suffix=0)
