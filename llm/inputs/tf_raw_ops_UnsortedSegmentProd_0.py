
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_unsorted_segment_prod_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "basic_example"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multidimensional data
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "multi_dim"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different data type (float)
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "float_data"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different segment IDs
    data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    segment_ids = np.array([0, 1, 0, 1, 2], dtype=np.int32)
    num_segments = np.array(3, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "diff_segment_ids"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: One segment
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 0], dtype=np.int32)
    num_segments = np.array(1, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "one_segment"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different num_segments value
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(5, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "diff_num_segments"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D data
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "3d_data"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64 segment_ids
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "int64_segment_ids"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int64 num_segments
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int64)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "int64_num_segments"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint8 data type
    data = np.array([1, 2, 3, 4], dtype=np.uint8)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": "uint8_data"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.UnsortedSegmentProd"] = tf_raw_ops_unsorted_segment_prod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.UnsortedSegmentProd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.UnsortedSegmentProd'.")

check_valid('tf.raw_ops.UnsortedSegmentProd', generated_inputs['tf.raw_ops.UnsortedSegmentProd'], lib="tf", suffix=0)
