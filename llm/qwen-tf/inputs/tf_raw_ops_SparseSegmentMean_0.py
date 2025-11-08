
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def generate_sparse_segment_mean_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32 data, int32 indices and segment_ids
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "test_1",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    # Input 2: With negative values in data
    data = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": True,
        "name": "test_2",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    # Input 3: Single dimension data with int64 indices and segment_ids
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    input_dict = {
        "sparse_gradient": False,
        "name": "test_3",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    # Input 4: Data with multiple dimensions and mixed segment_ids
    data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "test_4",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    # Input 5: With repeated segments and different indices
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": True,
        "name": "test_5",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    # Input 6: Data with float64 type and different indices
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "test_6",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    # Input 7: Single dimension with different number of elements
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int64)
    segment_ids = np.array([0, 1], dtype=np.int64)
    input_dict = {
        "sparse_gradient": True,
        "name": "test_7",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    # Input 8: Mixed data types and dimensions
    data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "test_8",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    # Input 9: Non-sequential segments
    data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    input_dict = {
        "sparse_gradient": True,
        "name": "test_9",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    # Input 10: Single dimension with different indices and segment_ids
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 1, 2], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "test_10",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSegmentMean"] = generate_sparse_segment_mean_inputs()

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
