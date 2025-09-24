
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_segment_min_inputs():
    list_of_inputs = []

    # Input 1: Basic example with int32 data and segment_ids
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "basic_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float32 data and int64 segment_ids
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "float32_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Data with multiple dimensions and int32 segment_ids
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "multi_dim_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Data with negative values and int32 segment_ids
    data = np.array([-1, -2, -3, -4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "negative_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single segment
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 0], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "single_segment"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: different data type: float64
    data = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: different data type: uint8
    data = np.array([1, 2, 3, 4], dtype=np.uint8)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "uint8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: different data type: int64
    data = np.array([1, 2, 3, 4], dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: different data type: bfloat16 (needs casting from float32)
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).astype(np.float16)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "bfloat16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Three dimensions
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "three dimensions"}
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
