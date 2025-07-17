
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sparse_slice_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    shape = np.array([2, 2], dtype=np.int64)
    start = np.array([0, 0], dtype=np.int64)
    size = np.array([2, 2], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "shape": shape, "start": start, "size": size, "name": "sparse_slice_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[0, 2], [0, 3], [1, 1]], dtype=np.int64)
    values = np.array([4, 5, 6], dtype=np.float32)
    shape = np.array([2, 4], dtype=np.int64)
    start = np.array([0, 2], dtype=np.int64)
    size = np.array([1, 2], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "shape": shape, "start": start, "size": size, "name": "sparse_slice_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([7, 8], dtype=np.int64)
    shape = np.array([3, 4], dtype=np.int64)
    start = np.array([0, 0], dtype=np.int64)
    size = np.array([2, 3], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "shape": shape, "start": start, "size": size, "name": "sparse_slice_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different values type
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([7.0, 8.0], dtype=np.float64)
    shape = np.array([3, 4], dtype=np.int64)
    start = np.array([0, 0], dtype=np.int64)
    size = np.array([2, 3], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "shape": shape, "start": start, "size": size, "name": "sparse_slice_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Start from non-zero
    indices = np.array([[2, 1], [2, 3]], dtype=np.int64)
    values = np.array([9, 10], dtype=np.int32)
    shape = np.array([4, 4], dtype=np.int64)
    start = np.array([2, 1], dtype=np.int64)
    size = np.array([1, 2], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "shape": shape, "start": start, "size": size, "name": "sparse_slice_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger shape
    indices = np.array([[0, 0], [1, 1], [2, 2], [3,3]], dtype=np.int64)
    values = np.array([11, 12, 13, 14], dtype=np.int32)
    shape = np.array([5, 5], dtype=np.int64)
    start = np.array([1, 1], dtype=np.int64)
    size = np.array([2, 2], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "shape": shape, "start": start, "size": size, "name": "sparse_slice_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: Different size
    indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    shape = np.array([4, 4], dtype=np.int64)
    start = np.array([0, 0], dtype=np.int64)
    size = np.array([3, 3], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "shape": shape, "start": start, "size": size, "name": "sparse_slice_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: start at index 1 for both dimensions
    indices = np.array([[1, 1], [2, 2]], dtype=np.int64)
    values = np.array([4, 5], dtype=np.int32)
    shape = np.array([4, 4], dtype=np.int64)
    start = np.array([1, 1], dtype=np.int64)
    size = np.array([2, 2], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "shape": shape, "start": start, "size": size, "name": "sparse_slice_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3x3 SparseTensor with slice [1,1] to [2,2]
    indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    shape = np.array([3, 3], dtype=np.int64)
    start = np.array([1, 1], dtype=np.int64)
    size = np.array([2, 2], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "shape": shape, "start": start, "size": size, "name": "sparse_slice_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Slice with different shape and start point
    indices = np.array([[0, 1], [1, 0], [2, 2]], dtype=np.int64)
    values = np.array([10, 20, 30], dtype=np.int32)
    shape = np.array([4, 4], dtype=np.int64)
    start = np.array([1, 0], dtype=np.int64)
    size = np.array([2, 3], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "shape": shape, "start": start, "size": size, "name": "sparse_slice_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseSlice"] = tf_raw_ops_sparse_slice_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseSlice' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSlice'.")

check_valid('tf.raw_ops.SparseSlice', generated_inputs['tf.raw_ops.SparseSlice'], lib="tf", suffix=0)
