
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sparse_segment_sum_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
    indices = np.array([0, 1, 2, 3]).astype(np.int32)
    segment_ids = np.array([0, 0, 1, 1]).astype(np.int32)
    sparse_gradient = False
    name = None
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]]).astype(np.int32)
    indices = np.array([0, 1]).astype(np.int32)
    segment_ids = np.array([0, 0]).astype(np.int32)
    sparse_gradient = True
    name = "sparse_segment_sum_example"
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]]).astype(np.float64)
    indices = np.array([0, 1]).astype(np.int64)
    segment_ids = np.array([0, 1]).astype(np.int64)
    sparse_gradient = False
    name = None
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]]).astype(np.uint8)
    indices = np.array([0, 1, 2]).astype(np.int32)
    segment_ids = np.array([0, 0, 1]).astype(np.int32)
    sparse_gradient = True
    name = "test_uint8"
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = np.array([1, 2, 3, 4, 5, 6]).astype(np.int64)
    indices = np.array([0, 2, 4]).astype(np.int64)
    segment_ids = np.array([0, 1, 2]).astype(np.int64)
    sparse_gradient = False
    name = None
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = np.array([[-1.0, 2.0], [3.0, -4.0], [-5.0, 6.0]]).astype(np.float32)
    indices = np.array([0, 1, 2]).astype(np.int32)
    segment_ids = np.array([0, 1, 1]).astype(np.int32)
    sparse_gradient = False
    name = None
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    data = np.array([1, 2, 3, 4]).astype(np.int32)
    indices = np.array([0, 1, 2, 3]).astype(np.int32)
    segment_ids = np.array([0, 0, 0, 0]).astype(np.int32)
    sparse_gradient = True
    name = "all_same_segment"
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    data = np.array([1, 2, 3, 4]).astype(np.float32)
    indices = np.array([0, 1]).astype(np.int32)
    segment_ids = np.array([0, 1]).astype(np.int32)
    sparse_gradient = False
    name = None
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    data = np.array([1, 2, 3, 4]).astype(np.float64)
    indices = np.array([0, 1, 2, 3]).astype(np.int64)
    segment_ids = np.array([0, 1, 2, 3]).astype(np.int64)
    sparse_gradient = True
    name = "distinct_segments"
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    data = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.int32)
    indices = np.array([0, 1, 2]).astype(np.int32)
    segment_ids = np.array([0, 0, 0]).astype(np.int32)
    sparse_gradient = False
    name = None
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)
    indices = np.array([0, 2, 4]).astype(np.int32)
    segment_ids = np.array([0, 1, 0]).astype(np.int32)
    sparse_gradient = False
    name = None
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)
    indices = np.array([0, 1, 2, 3, 4, 5]).astype(np.int32)
    segment_ids = np.array([0, 0, 1, 1, 2, 2]).astype(np.int32)
    sparse_gradient = True
    name = "test_sparse_gradient"
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]]).astype(np.float32)
    indices = np.array([0, 2]).astype(np.int32)
    segment_ids = np.array([0, 1]).astype(np.int32)
    sparse_gradient = False
    name = None
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseSegmentSum"] = tf_raw_ops_sparse_segment_sum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseSegmentSum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSegmentSum'.")

check_valid('tf.raw_ops.SparseSegmentSum', generated_inputs['tf.raw_ops.SparseSegmentSum'], lib="tf", suffix=0)
