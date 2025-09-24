
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def get_tf_raw_ops_sparse_segment_sqrt_n_grad_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.SparseSegmentSqrtNGrad function.
    """
    list_of_inputs = []

    # Input 1: Basic case with float32 and int32
    list_of_inputs.append(
        {
            "name": "basic_float32",
            "grad": np.random.rand(2, 3).astype(np.float32),
            "indices": np.array([0, 2, 1, 4], dtype=np.int32),
            "segment_ids": np.array([0, 0, 1, 1], dtype=np.int32),
            "output_dim0": np.array(5, dtype=np.int32),
        }
    )

    # Input 2: float64 grad and int64 indices/segments
    list_of_inputs.append(
        {
            "name": "float64_int64",
            "grad": np.random.rand(3, 2, 2).astype(np.float64),
            "indices": np.array([1, 5, 2, 8, 9], dtype=np.int64),
            "segment_ids": np.array([0, 0, 1, 2, 2], dtype=np.int64),
            "output_dim0": np.array(10, dtype=np.int32),
        }
    )

    # Input 3: 1D grad and a skipped segment ID
    list_of_inputs.append(
        {
            "name": "1d_grad_skipped_segment",
            "grad": np.random.rand(4).astype(np.float32),
            "indices": np.array([1, 2, 0, 3], dtype=np.int32),
            "segment_ids": np.array([0, 0, 2, 3], dtype=np.int32),
            "output_dim0": np.array(5, dtype=np.int32),
        }
    )

    # Input 4: Single segment
    list_of_inputs.append(
        {
            "name": "single_segment",
            "grad": np.random.rand(1, 4).astype(np.float32),
            "indices": np.array([3, 1, 0, 2], dtype=np.int32),
            "segment_ids": np.array([0, 0, 0, 0], dtype=np.int32),
            "output_dim0": np.array(5, dtype=np.int32),
        }
    )

    # Input 5: Higher dimensional grad
    list_of_inputs.append(
        {
            "name": "high_dim_grad",
            "grad": np.random.rand(2, 2, 2, 2).astype(np.float32),
            "indices": np.array([0, 1, 2, 3, 4], dtype=np.int64),
            "segment_ids": np.array([0, 0, 1, 1, 1], dtype=np.int64),
            "output_dim0": np.array(10, dtype=np.int32),
        }
    )

    # Input 6: float16 (half) grad type
    list_of_inputs.append(
        {
            "name": "float16_type",
            "grad": np.random.rand(3, 2).astype(np.float16),
            "indices": np.array([5, 1, 0, 4, 2, 3], dtype=np.int64),
            "segment_ids": np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
            "output_dim0": np.array(6, dtype=np.int32),
        }
    )

    # Input 7: Large output_dim0
    list_of_inputs.append(
        {
            "name": "large_output_dim0",
            "grad": np.random.rand(2, 2).astype(np.float32),
            "indices": np.array([10, 50, 25, 99], dtype=np.int32),
            "segment_ids": np.array([0, 0, 1, 1], dtype=np.int32),
            "output_dim0": np.array(100, dtype=np.int32),
        }
    )

    # Input 8: Empty inputs (0 segments, 0 indices)
    list_of_inputs.append(
        {
            "name": "empty_indices_segments",
            "grad": np.empty((0, 5), dtype=np.float32),
            "indices": np.empty(0, dtype=np.int32),
            "segment_ids": np.empty(0, dtype=np.int32),
            "output_dim0": np.array(10, dtype=np.int32),
        }
    )
    
    # Input 9: Negative values in grad
    list_of_inputs.append(
        {
            "name": "negative_grad",
            "grad": -np.random.rand(2, 3).astype(np.float32),
            "indices": np.array([0, 2, 1, 4], dtype=np.int32),
            "segment_ids": np.array([0, 0, 1, 1], dtype=np.int32),
            "output_dim0": np.array(5, dtype=np.int32),
        }
    )
    
    # Input 10: All indices map to the last segment
    list_of_inputs.append(
        {
            "name": "all_indices_last_segment",
            "grad": np.random.rand(3, 2).astype(np.float32),
            "indices": np.array([0, 1, 2, 3], dtype=np.int32),
            "segment_ids": np.array([2, 2, 2, 2], dtype=np.int32),
            "output_dim0": np.array(5, dtype=np.int32),
        }
    )

    # Input 11: output_dim0 is the same as the max index + 1
    list_of_inputs.append(
        {
            "name": "tight_output_dim0",
            "grad": np.random.rand(2, 2).astype(np.float64),
            "indices": np.array([3, 0, 2, 1], dtype=np.int64),
            "segment_ids": np.array([0, 0, 1, 1], dtype=np.int64),
            "output_dim0": np.array(4, dtype=np.int32),
        }
    )
    
    # Input 12: non-sorted segment_ids (valid as per TF docs)
    list_of_inputs.append(
        {
            "name": "non_sorted_segments",
            "grad": np.random.rand(3, 2).astype(np.float32),
            "indices": np.array([0, 2, 1], dtype=np.int32),
            "segment_ids": np.array([1, 0, 1], dtype=np.int32),
            "output_dim0": np.array(4, dtype=np.int32),
        }
    )
    
    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSegmentSqrtNGrad"] = get_tf_raw_ops_sparse_segment_sqrt_n_grad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseSegmentSqrtNGrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSegmentSqrtNGrad'.")

check_valid('tf.raw_ops.SparseSegmentSqrtNGrad', generated_inputs['tf.raw_ops.SparseSegmentSqrtNGrad'], lib="tf", suffix=0)
