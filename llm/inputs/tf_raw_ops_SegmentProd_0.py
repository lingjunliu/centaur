
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_segment_prod_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    name = "segment_prod_1"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    segment_ids = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    name = "segment_prod_2"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    name = "segment_prod_3"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    segment_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    name = "segment_prod_4"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = np.array([1, 2, 3, 4], dtype=np.float64)
    segment_ids = np.array([0, 0, 0, 0], dtype=np.int32)
    name = "segment_prod_5"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    name = "segment_prod_6"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    name = "segment_prod_7"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    data = np.array([2, 3, 4, 5], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    name = "segment_prod_8"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9 - uint8 type
    data = np.array([1, 2, 3, 4], dtype=np.uint8)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    name = "segment_prod_9"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - empty segment
    data = np.array([1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 2, 2], dtype=np.int32)
    name = "segment_prod_10"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    data = np.array([1j, 2j, 3j, 4j], dtype=np.complex64)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    name = "segment_prod_11"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SegmentProd"] = tf_raw_ops_segment_prod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SegmentProd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SegmentProd'.")

check_valid('tf.raw_ops.SegmentProd', generated_inputs['tf.raw_ops.SegmentProd'], lib="tf", suffix=0)
