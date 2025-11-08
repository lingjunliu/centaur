
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_SparseSegmentMean_inputs():
    list_of_inputs = []

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    sparse_gradient = False
    name = "test1"
    input_dict = {
        'sparse_gradient': sparse_gradient,
        'name': name,
        'data': data,
        'indices': indices,
        'segment_ids': segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float64)
    indices = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    sparse_gradient = True
    name = "test2"
    input_dict = {
        'sparse_gradient': sparse_gradient,
        'name': name,
        'data': data,
        'indices': indices,
        'segment_ids': segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float16)
    indices = np.array([0, 1, 0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    sparse_gradient = False
    name = "test3"
    input_dict = {
        'sparse_gradient': sparse_gradient,
        'name': name,
        'data': data,
        'indices': indices,
        'segment_ids': segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
    sparse_gradient = True
    name = "test4"
    input_dict = {
        'sparse_gradient': sparse_gradient,
        'name': name,
        'data': data,
        'indices': indices,
        'segment_ids': segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    sparse_gradient = False
    name = "test5"
    input_dict = {
        'sparse_gradient': sparse_gradient,
        'name': name,
        'data': data,
        'indices': indices,
        'segment_ids': segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSegmentMean"] = tf_raw_ops_SparseSegmentMean_inputs()

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
