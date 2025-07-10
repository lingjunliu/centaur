
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_unsorted_segment_sqrt_n_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
    segment_ids = np.array([0, 0, 1, 1]).astype(np.int32)
    num_segments = np.array(2).astype(np.int32)
    name = "segment_sqrt_n_1"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).astype(np.float32)
    segment_ids = np.array([0, 0, 1, 1]).astype(np.int32)
    num_segments = np.array(2).astype(np.int32)
    name = "segment_sqrt_n_2"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
    segment_ids = np.array([0, 1, 0, 1]).astype(np.int32)
    num_segments = np.array(3).astype(np.int32)
    name = "segment_sqrt_n_3"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).astype(np.float32)
    segment_ids = np.array([0, 1, 0, 1]).astype(np.int32)
    num_segments = np.array(3).astype(np.int32)
    name = "segment_sqrt_n_4"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)
    segment_ids = np.array([0, 0, 1, 1, 2]).astype(np.int32)
    num_segments = np.array(3).astype(np.int32)
    name = "segment_sqrt_n_5"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]).astype(np.float32)
    segment_ids = np.array([0, 0, 1, 1, 2]).astype(np.int32)
    num_segments = np.array(3).astype(np.int32)
    name = "segment_sqrt_n_6"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    data = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
    segment_ids = np.array([0, 0, 1, 0]).astype(np.int32)
    num_segments = np.array(2).astype(np.int32)
    name = "segment_sqrt_n_7"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).astype(np.float32)
    segment_ids = np.array([0, 0, 1, 0]).astype(np.int32)
    num_segments = np.array(2).astype(np.int32)
    name = "segment_sqrt_n_8"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    data = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
    segment_ids = np.array([1, 1, 0, 0]).astype(np.int32)
    num_segments = np.array(2).astype(np.int32)
    name = "segment_sqrt_n_9"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).astype(np.float32)
    segment_ids = np.array([1, 1, 0, 0]).astype(np.int32)
    num_segments = np.array(2).astype(np.int32)
    name = "segment_sqrt_n_10"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.unsorted_segment_sqrt_n"] = tf_math_unsorted_segment_sqrt_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.unsorted_segment_sqrt_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.unsorted_segment_sqrt_n'.")

check_valid('tf.math.unsorted_segment_sqrt_n', generated_inputs['tf.math.unsorted_segment_sqrt_n'], lib="tf", suffix=0)
