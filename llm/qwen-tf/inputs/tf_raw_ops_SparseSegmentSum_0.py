
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_segment_sum_inputs():
    list_of_inputs = []
    
    # Input 1, valid - 2D tensor with 3 rows
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid - 2D tensor with 3 rows
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid - 2D tensor with 3 rows
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid - 3D tensor with 2 rows
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid - 3D tensor with 3 rows
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 1, 2], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - 1D tensor with 4 elements
    data = np.array([1, 2, 3, 4], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - 2D tensor with 4 rows
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 1, 2, 3], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - 2D tensor with 3 rows, negative values
    data = np.array([[-1, -2, -3], [-4, -5, -6], [7, 8, 9]], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - 2D tensor with 3 rows, different dtypes
    data = np.array([[1.5, 2.7, 3.8], [4.1, 5.2, 6.3], [7.9, 8.1, 9.2]], dtype=np.float64)
    indices = np.array([0, 1], dtype=np.int64)
    segment_ids = np.array([0, 1], dtype=np.int64)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - 2D tensor with 2 rows, sparse gradient = True
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": True,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSegmentSum"] = tf_sparse_segment_sum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseSegmentSum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSegmentSum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseSegmentSum', generated_inputs['tf.raw_ops.SparseSegmentSum'], lib="tf", suffix=0)
