
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_segment_min_inputs():
    list_of_inputs = []

    # Input 1: Simple case with int32 data and segment_ids
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 data with different segment_ids
    data = np.array([5.0, 2.0, 8.0, 1.0], dtype=np.float32)
    segment_ids = np.array([0, 1, 0, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int64 data with repeated segment_ids
    data = np.array([10, 5, 12, 3], dtype=np.int64)
    segment_ids = np.array([0, 0, 0, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 data with larger segment_ids
    data = np.array([2.5, 1.5, 3.5, 0.5], dtype=np.float64)
    segment_ids = np.array([0, 1, 2, 0], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D data with int32 segment_ids
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint8 data with segment_ids
    data = np.array([255, 128, 64, 32], dtype=np.uint8)
    segment_ids = np.array([0, 1, 1, 2], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int16 data with int64 segment_ids
    data = np.array([-10, -5, 0, 5], dtype=np.int16)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half data
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty Segment
    data = np.array([1, 2, 3], dtype=np.int32)
    segment_ids = np.array([1, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Unsorted segment ids (CPU will error, but this is for GPU testing)
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([1, 0, 1, 0], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SegmentMin"] = tf_raw_ops_segment_min_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SegmentMin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SegmentMin'.")

check_valid('tf.raw_ops.SegmentMin', generated_inputs['tf.raw_ops.SegmentMin'], lib="tf", suffix=0)
