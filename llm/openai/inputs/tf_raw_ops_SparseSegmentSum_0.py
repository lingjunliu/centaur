
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_segment_sum_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array([[1, 2, 3, 4],
                     [-1, -2, -3, -4],
                     [5, 6, 7, 8]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "basic_one_segment",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([[1, 2, 3, 4],
                     [-1, -2, -3, -4],
                     [5, 6, 7, 8]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "two_segments",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array([[1, 2, 3, 4],
                     [-1, -2, -3, -4],
                     [5, 6, 7, 8]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": True,
        "name": "select_all_two_segments",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array([-3, -2, -1, 0, 1], dtype=np.int32)
    indices = np.array([0, 2, 4], dtype=np.int64)
    segment_ids = np.array([0, 1, 1], dtype=np.int64)
    input_dict = {
        "sparse_gradient": False,
        "name": "int32_vector_multi_segments",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
            [[13, 14], [15, 16]],
        ],
        dtype=np.uint8
    )
    indices = np.array([0, 3, 1, 1], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 2], dtype=np.int32)
    input_dict = {
        "sparse_gradient": True,
        "name": "uint8_3d_var_segments",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = np.array(
        [
            [1.5, -2.0, 3.25],
            [-4.5, 5.5, -6.75],
            [7.125, -8.875, 9.0],
            [0.5, 0.25, -0.75],
            [2.5, -3.5, 4.5],
            [-1.25, 2.0, -3.0]
        ],
        dtype=np.float64
    )
    indices = np.array([5, 3, 0, 2], dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    input_dict = {
        "sparse_gradient": False,
        "name": "float64_matrix_two_segments",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    data = np.array(
        [
            [[[1, -1], [2, -2]], [[3, -3], [4, -4]]],
            [[[5, -5], [6, -6]], [[7, -7], [8, -8]]],
            [[[9, -9], [10, -10]], [[11, -11], [12, -12]]]
        ],
        dtype=np.int64
    )
    indices = np.array([0, 2, 2], dtype=np.int32)
    segment_ids = np.array([1, 1, 5], dtype=np.int32)
    input_dict = {
        "sparse_gradient": True,
        "name": "int64_4d_with_gapped_segments",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (int8)
    data = np.array([[-128, -1, 0, 1, 127],
                     [10, -20, 30, -40, 50]], dtype=np.int8)
    indices = np.array([1, 0], dtype=np.int32)
    segment_ids = np.array([3, 3], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "int8_two_rows_single_segment_id3",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    data = np.array(
        [
            [0.5, -1.0, 1.5, -2.0],
            [2.5, -3.0, 3.5, -4.0],
            [4.5, -5.0, 5.5, -6.0],
            [6.5, -7.0, 7.5, -8.0],
            [8.5, -9.0, 9.5, -10.0]
        ],
        dtype=np.float16
    )
    indices = np.array([0, 4, 2, 1, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": True,
        "name": "float16_multi_segments_more_pairs",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (replace previous uint16 with int32)
    data = np.array([[[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]]], dtype=np.int32)
    indices = np.array([0], dtype=np.int32)
    segment_ids = np.array([0], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "int32_single_row_single_segment",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (replace previous uint32 with float32)
    data = np.array([10.0, 20.0, -30.0, 40.0, -50.0, 60.0, -70.0], dtype=np.float32)
    indices = np.array([1, 1, 6, 0], dtype=np.int64)
    segment_ids = np.array([0, 1, 1, 3], dtype=np.int64)
    input_dict = {
        "sparse_gradient": False,
        "name": "float32_vector_with_duplicates",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (replace previous uint64 with int16)
    data = (np.arange(15, dtype=np.int16) - 7).reshape(3, 5)
    indices = np.array([2, 0, 1], dtype=np.int32)
    segment_ids = np.array([0, 2, 2], dtype=np.int32)
    input_dict = {
        "sparse_gradient": True,
        "name": "int16_matrix_nonconsecutive_segments",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13
    data = np.array([-1000, 2000, -3000, 4000], dtype=np.int16)
    indices = np.array([0, 3, 2, 2, 1], dtype=np.int64)
    segment_ids = np.array([0, 1, 1, 1, 3], dtype=np.int64)
    input_dict = {
        "sparse_gradient": False,
        "name": "int16_vector_three_segments",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14
    data = np.arange(3*1*2*1*2, dtype=np.float32).reshape(3, 1, 2, 1, 2) - 5.0
    indices = np.array([2, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 2, 2], dtype=np.int32)
    input_dict = {
        "sparse_gradient": True,
        "name": "float32_5d_gapped_segments",
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
