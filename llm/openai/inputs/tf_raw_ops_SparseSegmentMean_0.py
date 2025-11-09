
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

np.random.seed(0)

def tf_sparse_segment_mean_inputs():
    list_of_inputs = []

    data = np.array([1.0, -2.5, 3.5, 4.0, 5.5, -6.0], dtype=np.float32)
    indices = np.array([0, 2, 5, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "case1",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([
        [1.0, 2.0, -1.0],
        [3.0, -4.0, 5.0],
        [6.0, 7.0, 8.0],
        [-9.0, 10.0, 11.0],
        [12.0, -13.0, 14.0]
    ], dtype=np.float64)
    indices = np.array([0, 1, 3, 4, 2], dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    input_dict = {
        "sparse_gradient": True,
        "name": "case2",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([
        [[1.0, -1.0], [2.0, -2.0]],
        [[3.0, -3.0], [4.0, -4.0]],
        [[5.0, -5.0], [6.0, -6.0]],
        [[7.0, -7.0], [8.0, -8.0]]
    ], dtype=np.float16)
    indices = np.array([3, 1, 0], dtype=np.int32)
    segment_ids = np.array([0, 0, 2], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "case3",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = (np.arange(8, dtype=np.float32).reshape(8, 1) * 0.5) - 2.0
    indices = np.array([7, 0, 2, 5, 3], dtype=np.int64)
    segment_ids = np.array([0, 1, 1, 3, 3], dtype=np.int64)
    input_dict = {
        "sparse_gradient": True,
        "name": "case4",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([10.0, -10.0, 20.0, -20.0], dtype=np.float64)
    indices = np.array([1, 1, 2, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "case5",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.random.randn(10, 4, 5).astype(np.float32)
    indices = np.arange(10, dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    input_dict = {
        "sparse_gradient": True,
        "name": "case6",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float16)
    indices = np.array([2, 2, 2], dtype=np.int32)
    segment_ids = np.array([0, 1, 2], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "case7",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = (np.arange(24, dtype=np.float32).reshape(6, 2, 1, 2) - 12.0) / 3.0
    indices = np.array([1, 1, 2, 4, 0], dtype=np.int64)
    segment_ids = np.array([0, 0, 2, 2, 5], dtype=np.int64)
    input_dict = {
        "sparse_gradient": True,
        "name": "case8",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([1.5, -2.5, 3.5, 4.5, -5.5], dtype=np.float16)
    indices = np.array([3], dtype=np.int32)
    segment_ids = np.array([3], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "case9",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[0.0, -1.0, 2.0, -3.0, 4.0]], dtype=np.float64)
    indices = np.array([0, 0, 0, 0], dtype=np.int64)
    segment_ids = np.array([0, 0, 0, 0], dtype=np.int64)
    input_dict = {
        "sparse_gradient": True,
        "name": "case10",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([
        [0.0, 1.0],
        [2.0, 3.0],
        [4.0, 5.0],
        [6.0, 7.0],
        [8.0, 9.0],
        [10.0, 11.0],
        [12.0, 13.0]
    ], dtype=np.float32)
    indices = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)
    segment_ids = np.array([0, 0, 2, 2, 2, 5, 5], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "case11",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([
        [[-1.0, 1.0], [-2.0, 2.0]],
        [[-3.0, 3.0], [-4.0, 4.0]]
    ], dtype=np.float32)
    indices = np.array([1, 1, 0], dtype=np.int64)
    segment_ids = np.array([0, 1, 1], dtype=np.int64)
    input_dict = {
        "sparse_gradient": True,
        "name": "case12",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSegmentMean"] = tf_sparse_segment_mean_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseSegmentMean' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSegmentMean'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseSegmentMean', generated_inputs['tf.raw_ops.SparseSegmentMean'], lib="tf", suffix=0)
