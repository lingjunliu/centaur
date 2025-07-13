
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_bincount_inputs():
    list_of_inputs = []

    # Input 1: Basic case, weights are None (treated as all 1s)
    arr = np.array([0, 1, 1, 2, 3, 2, 1, 0]).astype(np.int32)
    size = np.array(5).astype(np.int32)
    weights = np.array([]).astype(np.float32)  # Empty array for None equivalent

    input_dict = {
        "arr": arr,
        "size": size,
        "weights": weights,
        "name": "bincount_basic"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With integer weights
    arr = np.array([0, 1, 1, 2, 3, 2, 1, 0]).astype(np.int32)
    size = np.array(5).astype(np.int32)
    weights = np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(np.int32)

    input_dict = {
        "arr": arr,
        "size": size,
        "weights": weights,
        "name": "bincount_int_weights"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With float weights
    arr = np.array([0, 1, 1, 2, 3, 2, 1, 0]).astype(np.int32)
    size = np.array(5).astype(np.int32)
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).astype(np.float32)

    input_dict = {
        "arr": arr,
        "size": size,
        "weights": weights,
        "name": "bincount_float_weights"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different size
    arr = np.array([0, 1, 1, 2, 3, 2, 1, 0]).astype(np.int32)
    size = np.array(3).astype(np.int32)
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).astype(np.float32)

    input_dict = {
        "arr": arr,
        "size": size,
        "weights": weights,
        "name": "bincount_different_size"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  arr with values equal to size - 1
    arr = np.array([0, 1, 2, 2, 1, 0]).astype(np.int32)
    size = np.array(3).astype(np.int32)
    weights = np.array([1, 2, 3, 4, 5, 6]).astype(np.int32)

    input_dict = {
        "arr": arr,
        "size": size,
        "weights": weights,
        "name": "bincount_size_minus_one"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: size = 0
    arr = np.array([0, 1, 2, 2, 1, 0]).astype(np.int32)
    size = np.array(0).astype(np.int32)
    weights = np.array([1, 2, 3, 4, 5, 6]).astype(np.int32)

    input_dict = {
        "arr": arr,
        "size": size,
        "weights": weights,
        "name": "bincount_size_zero"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  arr with values exceeding size
    arr = np.array([0, 1, 2, 3, 4, 5]).astype(np.int32)
    size = np.array(3).astype(np.int32)
    weights = np.array([1, 2, 3, 4, 5, 6]).astype(np.int32)

    input_dict = {
        "arr": arr,
        "size": size,
        "weights": weights,
        "name": "bincount_exceeding_size"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: arr is empty
    arr = np.array([]).astype(np.int32)
    size = np.array(5).astype(np.int32)
    weights = np.array([]).astype(np.float32)

    input_dict = {
        "arr": arr,
        "size": size,
        "weights": weights,
        "name": "bincount_arr_empty"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: arr with larger values, int64 weights
    arr = np.array([0, 1, 5, 2, 1, 0, 8]).astype(np.int32)
    size = np.array(10).astype(np.int32)
    weights = np.array([1, 2, 3, 4, 5, 6, 7]).astype(np.int64)

    input_dict = {
        "arr": arr,
        "size": size,
        "weights": weights,
        "name": "bincount_larger_values_int64_weights"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: size 1
    arr = np.array([0, 0, 0]).astype(np.int32)
    size = np.array(1).astype(np.int32)
    weights = np.array([1, 2, 3]).astype(np.int32)

    input_dict = {
        "arr": arr,
        "size": size,
        "weights": weights,
        "name": "bincount_size_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: weights all zeros
    arr = np.array([0, 1, 2]).astype(np.int32)
    size = np.array(3).astype(np.int32)
    weights = np.array([0, 0, 0]).astype(np.int32)

    input_dict = {
        "arr": arr,
        "size": size,
        "weights": weights,
        "name": "bincount_weights_all_zeros"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Float64 weights
    arr = np.array([0, 1, 1, 2, 3, 2, 1, 0]).astype(np.int32)
    size = np.array(5).astype(np.int32)
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).astype(np.float64)

    input_dict = {
        "arr": arr,
        "size": size,
        "weights": weights,
        "name": "bincount_float64_weights"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    generated_inputs = {}
    generated_inputs["tf.raw_ops.Bincount"] = list_of_inputs

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Bincount' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Bincount'.")

check_valid('tf.raw_ops.Bincount', generated_inputs['tf.raw_ops.Bincount'], lib="tf", suffix=0)
