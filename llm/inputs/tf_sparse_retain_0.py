
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_retain_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    indices = np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    shape = np.array([3, 4], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    to_retain = np.array([True, False, True], dtype=np.bool_)
    input_dict = {"sp_input": sp_input, "to_retain": to_retain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: All true
    indices = np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    shape = np.array([3, 4], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    to_retain = np.array([True, True, True], dtype=np.bool_)
    input_dict = {"sp_input": sp_input, "to_retain": to_retain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: All false
    indices = np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    shape = np.array([3, 4], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    to_retain = np.array([False, False, False], dtype=np.bool_)
    input_dict = {"sp_input": sp_input, "to_retain": to_retain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D SparseTensor
    indices = np.array([[0], [2], [3]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    shape = np.array([5], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    to_retain = np.array([True, False, True], dtype=np.bool_)
    input_dict = {"sp_input": sp_input, "to_retain": to_retain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different value type
    indices = np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = np.array([3, 4], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    to_retain = np.array([True, False, True], dtype=np.bool_)
    input_dict = {"sp_input": sp_input, "to_retain": to_retain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: bool values
    indices = np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int64)
    values = np.array([True, False, True], dtype=np.bool_)
    shape = np.array([3, 4], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    to_retain = np.array([True, False, True], dtype=np.bool_)
    input_dict = {"sp_input": sp_input, "to_retain": to_retain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: string values
    indices = np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int64)
    values = np.array(["a", "b", "c"], dtype=np.string_)
    shape = np.array([3, 4], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    to_retain = np.array([True, False, True], dtype=np.bool_)
    input_dict = {"sp_input": sp_input, "to_retain": to_retain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: more indices, more values
    indices = np.array([[0, 0], [0, 1], [1, 2], [2, 3], [2, 2]], dtype=np.int64)
    values = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    shape = np.array([3, 4], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    to_retain = np.array([True, False, True, True, False], dtype=np.bool_)
    input_dict = {"sp_input": sp_input, "to_retain": to_retain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sparse.retain"] = tf_sparse_retain_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.retain' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.retain'.")

check_valid('tf.sparse.retain', generated_inputs['tf.sparse.retain'], lib="tf", suffix=0)
