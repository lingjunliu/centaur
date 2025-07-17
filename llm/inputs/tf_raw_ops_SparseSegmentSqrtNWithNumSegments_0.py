
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseSegmentSqrtNWithNumSegments_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    sparse_gradient = False
    name = None

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "sparse_gradient": sparse_gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    sparse_gradient = True
    name = "test2"

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "sparse_gradient": sparse_gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    indices = np.array([0, 1, 4, 5], dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    num_segments = np.array(2, dtype=np.int64)
    sparse_gradient = False
    name = None

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "sparse_gradient": sparse_gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
    indices = np.array([0, 2, 4], dtype=np.int32)
    segment_ids = np.array([0, 1, 1], dtype=np.int32)
    num_segments = np.array(3, dtype=np.int32)
    sparse_gradient = False
    name = None

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "sparse_gradient": sparse_gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(3, dtype=np.int32)
    sparse_gradient = False
    name = None

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "sparse_gradient": sparse_gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(4, dtype=np.int32)
    sparse_gradient = True
    name = "test6"

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "sparse_gradient": sparse_gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    indices = np.array([0, 1, 4, 5], dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    num_segments = np.array(3, dtype=np.int64)
    sparse_gradient = False
    name = None

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "sparse_gradient": sparse_gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
    indices = np.array([0, 2, 4], dtype=np.int32)
    segment_ids = np.array([0, 1, 1], dtype=np.int32)
    num_segments = np.array(4, dtype=np.int32)
    sparse_gradient = False
    name = None

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "sparse_gradient": sparse_gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(5, dtype=np.int32)
    sparse_gradient = False
    name = None

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "sparse_gradient": sparse_gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(5, dtype=np.int32)
    sparse_gradient = True
    name = "test10"

    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "sparse_gradient": sparse_gradient,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseSegmentSqrtNWithNumSegments"] = tf_raw_ops_SparseSegmentSqrtNWithNumSegments_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseSegmentSqrtNWithNumSegments' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSegmentSqrtNWithNumSegments'.")

check_valid('tf.raw_ops.SparseSegmentSqrtNWithNumSegments', generated_inputs['tf.raw_ops.SparseSegmentSqrtNWithNumSegments'], lib="tf", suffix=0)
