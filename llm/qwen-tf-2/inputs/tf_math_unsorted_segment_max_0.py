
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_unsorted_segment_max_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]], dtype=np.int32)
    segment_ids = np.array([0, 1, 0], dtype=np.int32)
    num_segments = 2
    name = "test1"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    segment_ids = np.array([0, 1], dtype=np.int64)
    num_segments = 2
    name = "test2"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    num_segments = 3
    name = "test3"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    segment_ids = np.array([0, 0, 1], dtype=np.int64)
    num_segments = 2
    name = "test4"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int16)
    segment_ids = np.array([0, 1], dtype=np.int32)
    num_segments = 2
    name = "test5"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    data = np.array([1.5, 2.7, 3.9, 4.1], dtype=np.float32)
    segment_ids = np.array([0, 1, 0, 1], dtype=np.int64)
    num_segments = 2
    name = "test6"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    num_segments = 2
    name = "test7"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    data = np.array([[-1, -2, -3], [4, 5, 6], [7, 8, 9]], dtype=np.int8)
    segment_ids = np.array([0, 1, 0], dtype=np.int64)
    num_segments = 2
    name = "test8"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    data = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=np.int16)
    segment_ids = np.array([0, 1], dtype=np.int64)
    num_segments = 2
    name = "test9"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float16)
    segment_ids = np.array([0, 1], dtype=np.int32)
    num_segments = 2
    name = "test10"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.unsorted_segment_max"] = tf_unsorted_segment_max_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.unsorted_segment_max' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.unsorted_segment_max'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.unsorted_segment_max', generated_inputs['tf.math.unsorted_segment_max'], lib="tf", suffix=0)
