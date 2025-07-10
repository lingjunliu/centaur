
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_unsorted_segment_min_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data type (float)
    data = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different segment IDs
    data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    segment_ids = np.array([0, 1, 0, 2, 1], dtype=np.int32)
    num_segments = np.array(3, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional data
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different number of segments
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 0], dtype=np.int32)
    num_segments = np.array(1, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger segment IDs
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    segment_ids = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    num_segments = np.array(3, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D data
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64
    data = np.array([1, 2, 3, 4], dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint8
    data = np.array([1, 2, 3, 4], dtype=np.uint8)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Empty segment
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 0], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.unsorted_segment_min"] = tf_math_unsorted_segment_min_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.unsorted_segment_min' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.unsorted_segment_min'.")

check_valid('tf.math.unsorted_segment_min', generated_inputs['tf.math.unsorted_segment_min'], lib="tf", suffix=0)
