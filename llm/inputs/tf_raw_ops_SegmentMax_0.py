
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_segment_max_inputs():
    list_of_inputs = []

    # Input 1: Basic example with int32 data and segment_ids
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 data and int64 segment_ids
    data = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype=np.float32)
    segment_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional data
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different segment_ids
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    segment_ids = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large segment_ids
    data = np.array([1, 2, 3], dtype=np.int32)
    segment_ids = np.array([100, 100, 200], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint8 data
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8)
    segment_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty segment
    data = np.array([1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 2], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bfloat16 data
    data = np.array([1.5, 2.5, 3.5], dtype=np.float16)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint32
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint32)
    segment_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    input_dict = {"data": data, "segment_ids": segment_ids, "name": "segment_max_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int8 with negative
    data = np.array([-1, 2, -3, 4, -5, 6], dtype=np.int8)
    segment_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
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
