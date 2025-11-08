
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_SparseSegmentSum_inputs():
    list_of_inputs = []

    data1 = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]], dtype=np.float32)
    indices1 = np.array([0, 1], dtype=np.int32)
    segment_ids1 = np.array([0, 0], dtype=np.int32)
    sparse_gradient1 = False
    name1 = "test1"

    input_dict1 = {
        "data": data1,
        "indices": indices1,
        "segment_ids": segment_ids1,
        "sparse_gradient": sparse_gradient1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    data2 = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]], dtype=np.int32)
    indices2 = np.array([0, 1, 2], dtype=np.int32)
    segment_ids2 = np.array([0, 0, 1], dtype=np.int32)
    sparse_gradient2 = True
    name2 = "test2"

    input_dict2 = {
        "data": data2,
        "indices": indices2,
        "segment_ids": segment_ids2,
        "sparse_gradient": sparse_gradient2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    data3 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    indices3 = np.array([0, 1], dtype=np.int32)
    segment_ids3 = np.array([0, 1], dtype=np.int32)
    sparse_gradient3 = False
    name3 = "test3"

    input_dict3 = {
        "data": data3,
        "indices": indices3,
        "segment_ids": segment_ids3,
        "sparse_gradient": sparse_gradient3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    data4 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    indices4 = np.array([0, 1], dtype=np.int32)
    segment_ids4 = np.array([0, 1], dtype=np.int32)
    sparse_gradient4 = False
    name4 = "test4"

    input_dict4 = {
        "data": data4,
        "indices": indices4,
        "segment_ids": segment_ids4,
        "sparse_gradient": sparse_gradient4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    data5 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    indices5 = np.array([0, 1], dtype=np.int32)
    segment_ids5 = np.array([0, 1], dtype=np.int32)
    sparse_gradient5 = True
    name5 = "test5"

    input_dict5 = {
        "data": data5,
        "indices": indices5,
        "segment_ids": segment_ids5,
        "sparse_gradient": sparse_gradient5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    data6 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices6 = np.array([0, 1], dtype=np.int32)
    segment_ids6 = np.array([0, 1], dtype=np.int32)
    sparse_gradient6 = False
    name6 = "test6"

    input_dict6 = {
        "data": data6,
        "indices": indices6,
        "segment_ids": segment_ids6,
        "sparse_gradient": sparse_gradient6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSegmentSum"] = tf_raw_ops_SparseSegmentSum_inputs()

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
