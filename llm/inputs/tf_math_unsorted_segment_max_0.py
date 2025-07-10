
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_unsorted_segment_max_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array([1, 2, 3, 4]).astype(np.int32)
    segment_ids = np.array([0, 0, 1, 1]).astype(np.int32)
    num_segments = 2
    name = "example1"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([[1,2,3,4], [5,6,7,8], [4,3,2,1]]).astype(np.int32)
    segment_ids = np.array([0, 1, 0]).astype(np.int32)
    num_segments = 2
    name = "example2"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
    segment_ids = np.array([0, 0, 1, 1]).astype(np.int32)
    num_segments = 2
    name = "example3"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array([1, 2, 3, 4, 5]).astype(np.int32)
    segment_ids = np.array([0, 0, 1, 1, 2]).astype(np.int32)
    num_segments = 3
    name = "example4"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.int32)
    segment_ids = np.array([0, 1, 0]).astype(np.int32)
    num_segments = 2
    name = "example5"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = np.array([1, 2, 3, 4]).astype(np.int64)
    segment_ids = np.array([0, 0, 1, 1]).astype(np.int64)
    num_segments = 2
    name = "example6"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    data = np.array([1, 2, 3, 4]).astype(np.float64)
    segment_ids = np.array([0, 0, 1, 1]).astype(np.int32)
    num_segments = 2
    name = "example7"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    data = np.array([[-1, -2], [-3, -4], [-5, -6]]).astype(np.int32)
    segment_ids = np.array([0, 1, 0]).astype(np.int32)
    num_segments = 2
    name = "example8"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    data = np.array([1, 2, 3, 4]).astype(np.int32)
    segment_ids = np.array([0, 0, 1, 0]).astype(np.int32)
    num_segments = 2
    name = "example9"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    data = np.array([1, 2, 3, 4]).astype(np.uint8)
    segment_ids = np.array([0, 0, 1, 1]).astype(np.int32)
    num_segments = 2
    name = "example10"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.unsorted_segment_max"] = tf_math_unsorted_segment_max_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.unsorted_segment_max' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.unsorted_segment_max'.")

check_valid('tf.math.unsorted_segment_max', generated_inputs['tf.math.unsorted_segment_max'], lib="tf", suffix=0)
