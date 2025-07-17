
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sparse_concat_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    indices1 = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values1 = np.array([1, 2], dtype=np.int32)
    shape1 = np.array([2, 3], dtype=np.int64)
    indices2 = np.array([[0, 1], [1, 0]], dtype=np.int64)
    values2 = np.array([3, 4], dtype=np.int32)
    shape2 = np.array([2, 3], dtype=np.int64)
    concat_dim = 1

    input_dict = {
        "indices": [indices1, indices2],
        "values": [values1, values2],
        "shapes": [shape1, shape2],
        "concat_dim": concat_dim,
        "name": "sparse_concat_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shapes along concat_dim
    indices1 = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values1 = np.array([1, 2], dtype=np.float32)
    shape1 = np.array([2, 3], dtype=np.int64)
    indices2 = np.array([[0, 1], [1, 0]], dtype=np.int64)
    values2 = np.array([3, 4], dtype=np.float32)
    shape2 = np.array([2, 4], dtype=np.int64)
    concat_dim = 1

    input_dict = {
        "indices": [indices1, indices2],
        "values": [values1, values2],
        "shapes": [shape1, shape2],
        "concat_dim": concat_dim,
        "name": "sparse_concat_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: concat_dim = 0
    indices1 = np.array([[0, 0], [0, 2]], dtype=np.int64)
    values1 = np.array([1, 2], dtype=np.int64)
    shape1 = np.array([1, 3], dtype=np.int64)
    indices2 = np.array([[0, 1], [0, 0]], dtype=np.int64)
    values2 = np.array([3, 4], dtype=np.int64)
    shape2 = np.array([1, 3], dtype=np.int64)
    concat_dim = 0

    input_dict = {
        "indices": [indices1, indices2],
        "values": [values1, values2],
        "shapes": [shape1, shape2],
        "concat_dim": concat_dim,
        "name": "sparse_concat_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: More SparseTensors
    indices1 = np.array([[0, 0]], dtype=np.int64)
    values1 = np.array([1], dtype=np.int32)
    shape1 = np.array([2, 3], dtype=np.int64)
    indices2 = np.array([[1, 1]], dtype=np.int64)
    values2 = np.array([2], dtype=np.int32)
    shape2 = np.array([2, 3], dtype=np.int64)

    concat_dim = 1

    input_dict = {
        "indices": [indices1, indices2],
        "values": [values1, values2],
        "shapes": [shape1, shape2],
        "concat_dim": concat_dim,
        "name": "sparse_concat_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
def check_valid(api, generated_inputs, lib="tf", suffix=0):
    pass

generated_inputs["tf.raw_ops.SparseConcat"] = tf_raw_ops_sparse_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseConcat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseConcat'.")

check_valid('tf.raw_ops.SparseConcat', generated_inputs['tf.raw_ops.SparseConcat'], lib="tf", suffix=0)
