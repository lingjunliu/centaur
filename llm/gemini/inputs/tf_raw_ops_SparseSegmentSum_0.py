
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseSegmentSum_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    sparse_gradient = False
    name = "sparse_segment_sum_1"
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    sparse_gradient = False
    name = "sparse_segment_sum_2"
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    sparse_gradient = False
    name = "sparse_segment_sum_3"
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    sparse_gradient = True
    name = "sparse_segment_sum_4"
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]], dtype=np.float64)
    indices = np.array([0, 1, 2], dtype=np.int64)
    segment_ids = np.array([0, 0, 1], dtype=np.int64)
    sparse_gradient = False
    name = "sparse_segment_sum_5"
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    indices = np.array([0, 2, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    sparse_gradient = True
    name = "sparse_segment_sum_6"
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    data = np.array([[-1, -2], [-3, -4]], dtype=np.int32)
    indices = np.array([0], dtype=np.int32)
    segment_ids = np.array([0], dtype=np.int32)
    sparse_gradient = False
    name = "sparse_segment_sum_7"
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    indices = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    sparse_gradient = True
    name = "sparse_segment_sum_8"
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    sparse_gradient = False
    name = "sparse_segment_sum_9"
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    sparse_gradient = True
    name = "sparse_segment_sum_10"
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int64)
    segment_ids = np.array([0, 1], dtype=np.int64)
    sparse_gradient = False
    name = "sparse_segment_sum_11"
    input_dict = {"data": data, "indices": indices, "segment_ids": segment_ids, "sparse_gradient": sparse_gradient, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseSegmentSum"] = tf_raw_ops_SparseSegmentSum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseSegmentSum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSegmentSum'.")

check_valid('tf.raw_ops.SparseSegmentSum', generated_inputs['tf.raw_ops.SparseSegmentSum'], lib="tf", suffix=0)
