
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_segment_sum_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    name = "segment_sum_1"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    name = "segment_sum_2"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    segment_ids = np.array([0, 1, 2, 2], dtype=np.int64)
    name = "segment_sum_3"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    segment_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    name = "segment_sum_4"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    name = "segment_sum_6"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = np.array([-1, -2, -3, -4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    name = "segment_sum_7"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    data = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]], dtype=np.float32)
    segment_ids = np.array([0, 1, 1], dtype=np.int32)
    name = "segment_sum_8"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    name = "segment_sum_9"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    data = np.array([1, 2, 3], dtype=np.int32)
    segment_ids = np.array([2, 2, 2], dtype=np.int32)
    name = "segment_sum_10"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    data = np.array([1], dtype=np.int32)
    segment_ids = np.array([0], dtype=np.int32)
    name = "segment_sum_11"
    input_dict = {"data": data, "segment_ids": segment_ids, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.segment_sum"] = tf_math_segment_sum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.segment_sum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.segment_sum'.")

check_valid('tf.math.segment_sum', generated_inputs['tf.math.segment_sum'], lib="tf", suffix=0)
