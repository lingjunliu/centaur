
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ragged_bincount_inputs():
    list_of_inputs = []

    # Input 1
    splits = np.array([0, 2, 5], dtype=np.int64)
    values = np.array([1, 2, 0, 1, 2], dtype=np.int32)
    size = np.array(5, dtype=np.int32)
    weights = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    binary_output = False
    name = "bincount_1"
    input_dict = {"splits": splits, "values": values, "size": size, "weights": weights, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    splits = np.array([0, 3], dtype=np.int64)
    values = np.array([0, 1, 2], dtype=np.int32)
    size = np.array(3, dtype=np.int32)
    weights = np.array([1, 2, 3], dtype=np.int32)
    binary_output = True
    name = "bincount_2"
    input_dict = {"splits": splits, "values": values, "size": size, "weights": weights, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    splits = np.array([0, 1, 2, 3], dtype=np.int64)
    values = np.array([0, 1, 2], dtype=np.int32)
    size = np.array(4, dtype=np.int32)
    weights = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    binary_output = False
    name = "bincount_3"
    input_dict = {"splits": splits, "values": values, "size": size, "weights": weights, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    splits = np.array([0, 2], dtype=np.int64)
    values = np.array([0, 1], dtype=np.int64)
    size = np.array(2, dtype=np.int64)
    weights = np.array([1, 2], dtype=np.int64)
    binary_output = True
    name = "bincount_4"
    input_dict = {"splits": splits, "values": values, "size": size, "weights": weights, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    splits = np.array([0, 1, 3], dtype=np.int64)
    values = np.array([0, 1, 2], dtype=np.int64)[:1]
    size = np.array(5, dtype=np.int64)
    weights = np.array([1.0], dtype=np.float32)
    binary_output = False
    name = "bincount_5"
    input_dict = {"splits": splits, "values": values, "size": size, "weights": weights, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    splits = np.array([0, 4], dtype=np.int64)
    values = np.array([0, 1, 2, 3], dtype=np.int32)
    size = np.array(6, dtype=np.int32)
    weights = np.array([1, 2, 3, 4], dtype=np.int32)
    binary_output = True
    name = "bincount_6"
    input_dict = {"splits": splits, "values": values, "size": size, "weights": weights, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    splits = np.array([0, 0], dtype=np.int64)
    values = np.array([], dtype=np.int32)
    size = np.array(5, dtype=np.int32)
    weights = np.array([], dtype=np.float32)
    binary_output = False
    name = "bincount_7"
    input_dict = {"splits": splits, "values": values, "size": size, "weights": weights, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    splits = np.array([0, 3], dtype=np.int64)
    values = np.array([2,1,0], dtype=np.int64)
    size = np.array(4, dtype=np.int64)
    weights = np.array([0.5, 0.25, 0.125], dtype=np.float64)
    binary_output = True
    name = "bincount_8"
    input_dict = {"splits": splits, "values": values, "size": size, "weights": weights, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    splits = np.array([0, 1, 2, 3], dtype=np.int64)
    values = np.array([0,1,2], dtype=np.int32)
    size = np.array(3, dtype=np.int32)
    weights = np.array([1,1,1], dtype=np.int32)
    binary_output = True
    name = "bincount_9"
    input_dict = {"splits": splits, "values": values, "size": size, "weights": weights, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    splits = np.array([0, 2, 4], dtype=np.int64)
    values = np.array([0, 1, 0, 1], dtype=np.int64)
    size = np.array(4, dtype=np.int64)
    weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    binary_output = False
    name = "bincount_10"
    input_dict = {"splits": splits, "values": values, "size": size, "weights": weights, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    splits = np.array([0, 0, 0], dtype=np.int64)
    values = np.array([], dtype=np.int32)
    size = np.array(3, dtype=np.int32)
    weights = np.array([], dtype=np.int32)
    binary_output = True
    name = "bincount_11"
    input_dict = {"splits": splits, "values": values, "size": size, "weights": weights, "binary_output": binary_output, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RaggedBincount' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RaggedBincount'.")

check_valid('tf.raw_ops.RaggedBincount', generated_inputs['tf.raw_ops.RaggedBincount'], lib="tf", suffix=0)
