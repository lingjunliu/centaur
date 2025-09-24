
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_segment_sum_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array([1, 2, 3, 4]).astype(np.float32)
    indices = np.array([0, 1, 2, 3]).astype(np.int32)
    segment_ids = np.array([0, 0, 1, 1]).astype(np.int32)
    num_segments = 2
    name = "segment_sum_1"
    sparse_gradient = False

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name,
        "sparse_gradient": sparse_gradient
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).astype(np.float32)
    indices = np.array([0, 1]).astype(np.int32)
    segment_ids = np.array([0, 1]).astype(np.int32)
    num_segments = 2
    name = "segment_sum_2"
    sparse_gradient = True

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name,
        "sparse_gradient": sparse_gradient
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]]).astype(np.float32)
    indices = np.array([0, 2, 3]).astype(np.int32)
    segment_ids = np.array([0, 0, 1]).astype(np.int32)
    num_segments = 2
    name = "segment_sum_3"
    sparse_gradient = False

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name,
        "sparse_gradient": sparse_gradient
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)
    indices = np.array([0, 2, 4]).astype(np.int32)
    segment_ids = np.array([0, 1, 1]).astype(np.int32)
    num_segments = 2
    name = "segment_sum_4"
    sparse_gradient = True

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name,
        "sparse_gradient": sparse_gradient
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: with missing segment ids
    data = np.array([[1, 2], [3, 4]]).astype(np.float32)
    indices = np.array([0, 1]).astype(np.int32)
    segment_ids = np.array([0, 2]).astype(np.int32)
    num_segments = 4
    name = "segment_sum_5"
    sparse_gradient = False

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name,
        "sparse_gradient": sparse_gradient
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: num_segments = None
    data = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
    indices = np.array([0, 1, 2]).astype(np.int32)
    segment_ids = np.array([0, 0, 1]).astype(np.int32)
    num_segments = None
    name = "segment_sum_6"
    sparse_gradient = True

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name,
        "sparse_gradient": sparse_gradient
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: different dtype for data
    data = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.int32)
    indices = np.array([0, 1, 2]).astype(np.int32)
    segment_ids = np.array([0, 0, 1]).astype(np.int32)
    num_segments = None
    name = "segment_sum_7"
    sparse_gradient = True

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name,
        "sparse_gradient": sparse_gradient
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: one element in segment_ids
    data = np.array([1, 2, 3, 4]).astype(np.float32)
    indices = np.array([0]).astype(np.int32)
    segment_ids = np.array([0]).astype(np.int32)
    num_segments = 1
    name = "segment_sum_8"
    sparse_gradient = False

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name,
        "sparse_gradient": sparse_gradient
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: more segments than indices
    data = np.array([1, 2]).astype(np.float32)
    indices = np.array([0]).astype(np.int32)
    segment_ids = np.array([0]).astype(np.int32)
    num_segments = 3
    name = "segment_sum_9"
    sparse_gradient = True

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name,
        "sparse_gradient": sparse_gradient
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: empty data
    data = np.array([]).astype(np.float32)
    indices = np.array([]).astype(np.int32)
    segment_ids = np.array([]).astype(np.int32)
    num_segments = 0
    name = "segment_sum_10"
    sparse_gradient = False

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name,
        "sparse_gradient": sparse_gradient
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.segment_sum"] = tf_sparse_segment_sum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.segment_sum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.segment_sum'.")

check_valid('tf.sparse.segment_sum', generated_inputs['tf.sparse.segment_sum'], lib="tf", suffix=0)
