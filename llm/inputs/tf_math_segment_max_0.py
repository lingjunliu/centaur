
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_segment_max_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    name = "segment_max_1"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multi-dimensional data
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    name = "segment_max_2"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single segment
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 0], dtype=np.int32)
    name = "segment_max_3"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data type (float32)
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    name = "segment_max_4"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different data type (int64)
    data = np.array([1, 2, 3, 4], dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    name = "segment_max_5"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty segment
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 2, 2], dtype=np.int32)
    name = "segment_max_6"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Segment IDs with a gap
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 3, 3], dtype=np.int32)
    name = "segment_max_7"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger segment IDs
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([10, 10, 11, 11], dtype=np.int32)
    name = "segment_max_8"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D data
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    name = "segment_max_9"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint8 data
    data = np.array([1, 2, 3, 4], dtype=np.uint8)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    name = "segment_max_10"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.segment_max"] = tf_math_segment_max_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.segment_max' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.segment_max'.")

check_valid('tf.math.segment_max', generated_inputs['tf.math.segment_max'], lib="tf", suffix=0)
