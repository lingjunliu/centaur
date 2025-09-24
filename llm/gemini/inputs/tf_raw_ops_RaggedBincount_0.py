
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ragged_bincount_inputs():
    list_of_inputs = []

    # Input 1: Basic case with int32 values and float32 weights
    splits = np.array([0, 3, 5], dtype=np.int64)
    values = np.array([0, 1, 2, 0, 1], dtype=np.int32)
    size = np.array(4, dtype=np.int32)
    weights = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    binary_output = False
    name = None

    input_dict = {
        "splits": splits,
        "values": values,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int64 values and int64 weights
    splits = np.array([0, 2, 4], dtype=np.int64)
    values = np.array([1, 5, 2, 3], dtype=np.int64)
    size = np.array(7, dtype=np.int64)
    weights = np.array([1, 2, 3, 4], dtype=np.int64)
    binary_output = True
    name = "bincount_int64"

    input_dict = {
        "splits": splits,
        "values": values,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty weights
    splits = np.array([0, 2], dtype=np.int64)
    values = np.array([0, 1], dtype=np.int32)
    size = np.array(3, dtype=np.int32)
    weights = np.array([], dtype=np.float32)
    binary_output = False
    name = None

    input_dict = {
        "splits": splits,
        "values": values,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 weights
    splits = np.array([0, 1, 3], dtype=np.int64)
    values = np.array([2, 1, 2], dtype=np.int32)
    size = np.array(5, dtype=np.int32)
    weights = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    binary_output = True
    name = None

    input_dict = {
        "splits": splits,
        "values": values,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger size
    splits = np.array([0, 4], dtype=np.int64)
    values = np.array([0, 1, 2, 3], dtype=np.int32)
    size = np.array(10, dtype=np.int32)
    weights = np.array([1, 1, 1, 1], dtype=np.int32)
    binary_output = False
    name = None

    input_dict = {
        "splits": splits,
        "values": values,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All values out of range
    splits = np.array([0, 3], dtype=np.int64)
    values = np.array([5, 6, 7], dtype=np.int32)
    size = np.array(4, dtype=np.int32)
    weights = np.array([1, 1, 1], dtype=np.float32)
    binary_output = False
    name = None

    input_dict = {
        "splits": splits,
        "values": values,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Mixed values in and out of range
    splits = np.array([0, 4], dtype=np.int64)
    values = np.array([0, 1, 5, 2], dtype=np.int32)
    size = np.array(4, dtype=np.int32)
    weights = np.array([1, 2, 3, 4], dtype=np.float32)
    binary_output = False
    name = None

    input_dict = {
        "splits": splits,
        "values": values,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: One split
    splits = np.array([0, 5], dtype=np.int64)
    values = np.array([0, 1, 2, 3, 0], dtype=np.int32)
    size = np.array(4, dtype=np.int32)
    weights = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    binary_output = True
    name = None

    input_dict = {
        "splits": splits,
        "values": values,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int64 values, float32 weights
    splits = np.array([0, 3], dtype=np.int64)
    values = np.array([0, 1, 2], dtype=np.int64)
    size = np.array(4, dtype=np.int64)
    weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    binary_output = False
    name = None

    input_dict = {
        "splits": splits,
        "values": values,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int32 values, int64 weights, binary_output=True
    splits = np.array([0, 2, 4], dtype=np.int64)
    values = np.array([0, 1, 0, 2], dtype=np.int32)
    size = np.array(3, dtype=np.int32)
    weights = np.array([1, 2, 3, 4], dtype=np.int64)
    binary_output = True
    name = None

    input_dict = {
        "splits": splits,
        "values": values,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Zero size, empty values and weights
    splits = np.array([0, 0], dtype=np.int64)
    values = np.array([], dtype=np.int32)
    size = np.array(0, dtype=np.int32)
    weights = np.array([], dtype=np.float32)
    binary_output = False
    name = None

    input_dict = {
        "splits": splits,
        "values": values,
        "size": size,
        "weights": weights,
        "binary_output": binary_output,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RaggedBincount"] = tf_raw_ops_ragged_bincount_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RaggedBincount' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RaggedBincount'.")

check_valid('tf.raw_ops.RaggedBincount', generated_inputs['tf.raw_ops.RaggedBincount'], lib="tf", suffix=0)
