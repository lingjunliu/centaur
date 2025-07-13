
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseSplit_inputs():
    list_of_inputs = []

    # Input 1
    split_dim = np.array(1, dtype=np.int64)
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.float32)
    shape = np.array([2, 3], dtype=np.int64)
    num_split = 2
    name = "split_1"

    input_dict = {
        'split_dim': split_dim,
        'indices': indices,
        'values': values,
        'shape': shape,
        'num_split': num_split,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    split_dim = np.array(0, dtype=np.int64)
    indices = np.array([[0, 0], [1, 2], [2, 1]], dtype=np.int64)
    values = np.array([4, 5, 6], dtype=np.int32)
    shape = np.array([3, 3], dtype=np.int64)
    num_split = 3
    name = "split_2"

    input_dict = {
        'split_dim': split_dim,
        'indices': indices,
        'values': values,
        'shape': shape,
        'num_split': num_split,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    split_dim = np.array(1, dtype=np.int64)
    indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 2]], dtype=np.int64)
    values = np.array([7, 8, 9], dtype=np.float64)
    shape = np.array([2, 2, 3], dtype=np.int64)
    num_split = 2
    name = "split_3"

    input_dict = {
        'split_dim': split_dim,
        'indices': indices,
        'values': values,
        'shape': shape,
        'num_split': num_split,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    split_dim = np.array(0, dtype=np.int64)
    indices = np.array([[0, 0], [1, 1], [2, 2], [3, 0]], dtype=np.int64)
    values = np.array([10, 11, 12, 13], dtype=np.int8)
    shape = np.array([4, 3], dtype=np.int64)
    num_split = 4
    name = "split_4"

    input_dict = {
        'split_dim': split_dim,
        'indices': indices,
        'values': values,
        'shape': shape,
        'num_split': num_split,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseSplit"] = tf_raw_ops_SparseSplit_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseSplit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSplit'.")

check_valid('tf.raw_ops.SparseSplit', generated_inputs['tf.raw_ops.SparseSplit'], lib="tf", suffix=0)
