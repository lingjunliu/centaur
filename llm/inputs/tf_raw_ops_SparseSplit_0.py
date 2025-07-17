
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
    indices = np.array([[0, 0], [0, 1], [1, 2]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.float32)
    shape = np.array([2, 3], dtype=np.int64)
    num_split = 2
    input_dict = {
        "split_dim": split_dim,
        "indices": indices,
        "values": values,
        "shape": shape,
        "num_split": num_split,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    split_dim = np.array(0, dtype=np.int64)
    indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    values = np.array([4, 5, 6], dtype=np.int32)
    shape = np.array([3, 3], dtype=np.int64)
    num_split = 3
    input_dict = {
        "split_dim": split_dim,
        "indices": indices,
        "values": values,
        "shape": shape,
        "num_split": num_split,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    split_dim = np.array(1, dtype=np.int64)
    indices = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values = np.array([7, 8, 9], dtype=np.float64)
    shape = np.array([2, 4], dtype=np.int64)
    num_split = 2
    input_dict = {
        "split_dim": split_dim,
        "indices": indices,
        "values": values,
        "shape": shape,
        "num_split": num_split,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 4
    split_dim = np.array(0, dtype=np.int64)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values = np.array([10, 11], dtype=np.int64)
    shape = np.array([2, 2], dtype=np.int64)
    num_split = 1
    input_dict = {
        "split_dim": split_dim,
        "indices": indices,
        "values": values,
        "shape": shape,
        "num_split": num_split,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    split_dim = np.array(1, dtype=np.int64)
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([12, 13, 14, 15], dtype=np.int64)
    shape = np.array([2, 2], dtype=np.int64)
    num_split = 2 # Reduced num_split
    input_dict = {
        "split_dim": split_dim,
        "indices": indices,
        "values": values,
        "shape": shape,
        "num_split": num_split,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    split_dim = np.array(0, dtype=np.int64)
    indices = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int64)
    values = np.array([16, 17, 18, 19], dtype=np.float32)
    shape = np.array([4, 4], dtype=np.int64)
    num_split = 2
    input_dict = {
        "split_dim": split_dim,
        "indices": indices,
        "values": values,
        "shape": shape,
        "num_split": num_split,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    split_dim = np.array(1, dtype=np.int64)
    indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    values = np.array([20, 21, 22], dtype=np.float64)
    shape = np.array([2, 3], dtype=np.int64)
    num_split = 3
    input_dict = {
        "split_dim": split_dim,
        "indices": indices,
        "values": values,
        "shape": shape,
        "num_split": num_split,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    split_dim = np.array(0, dtype=np.int64)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values = np.array([23, 24], dtype=np.int32)
    shape = np.array([2, 2], dtype=np.int64)
    num_split = 2
    input_dict = {
        "split_dim": split_dim,
        "indices": indices,
        "values": values,
        "shape": shape,
        "num_split": num_split,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    split_dim = np.array(0, dtype=np.int64)
    indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    values = np.array([25, 26, 27], dtype=np.int64)
    shape = np.array([3, 3], dtype=np.int64)
    num_split = 1
    input_dict = {
        "split_dim": split_dim,
        "indices": indices,
        "values": values,
        "shape": shape,
        "num_split": num_split,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    split_dim = np.array(1, dtype=np.int64)
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2,0]], dtype=np.int64)
    values = np.array([28, 29, 30, 31, 32], dtype=np.float32)
    shape = np.array([3, 2], dtype=np.int64)
    num_split = 2
    input_dict = {
        "split_dim": split_dim,
        "indices": indices,
        "values": values,
        "shape": shape,
        "num_split": num_split,
        "name": None
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
