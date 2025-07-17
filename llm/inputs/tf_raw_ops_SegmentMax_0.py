
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_segment_max_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data type (float32)
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: More segments
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multiple identical segment IDs
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different data type (int64) and segment_ids (int64)
    data = np.array([1, 2, 3, 4], dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty segment
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 2, 2], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: multi-dimensional data
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different data type (float64) and larger values
    data = np.array([1e9, 2e9, 3e9, 4e9], dtype=np.float64)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Empty input array
    data = np.array([], dtype=np.int32)
    segment_ids = np.array([], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Gaps in segment IDs
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 2, 2], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SegmentMax"] = tf_raw_ops_segment_max_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SegmentMax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SegmentMax'.")

check_valid('tf.raw_ops.SegmentMax', generated_inputs['tf.raw_ops.SegmentMax'], lib="tf", suffix=0)
