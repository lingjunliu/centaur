
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_bincount_inputs():
    list_of_inputs = []

    # Input 1: Basic case with weights = None
    arr = np.array([1, 2, 3, 0, 1, 2]).astype(np.int32)
    size = np.array(5).astype(np.int32)
    weights = np.array([]).astype(np.float32)
    input_dict = {"name": "bincount_1", "arr": arr, "size": size, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With integer weights
    arr = np.array([1, 2, 3, 0, 1, 2]).astype(np.int32)
    size = np.array(5).astype(np.int32)
    weights = np.array([1, 2, 3, 4, 5, 6]).astype(np.int32)
    input_dict = {"name": "bincount_2", "arr": arr, "size": size, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With float weights
    arr = np.array([1, 2, 3, 0, 1, 2]).astype(np.int32)
    size = np.array(5).astype(np.int32)
    weights = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]).astype(np.float32)
    input_dict = {"name": "bincount_3", "arr": arr, "size": size, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: arr with zero size
    arr = np.array([]).astype(np.int32)
    size = np.array(5).astype(np.int32)
    weights = np.array([]).astype(np.float32)
    input_dict = {"name": "bincount_4", "arr": arr, "size": size, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: weights with int64
    arr = np.array([1, 2, 3, 0, 1, 2]).astype(np.int32)
    size = np.array(5).astype(np.int32)
    weights = np.array([1, 2, 3, 4, 5, 6]).astype(np.int64)
    input_dict = {"name": "bincount_5", "arr": arr, "size": size, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: weights with float64
    arr = np.array([1, 2, 3, 0, 1, 2]).astype(np.int32)
    size = np.array(5).astype(np.int32)
    weights = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]).astype(np.float64)
    input_dict = {"name": "bincount_6", "arr": arr, "size": size, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: size = 0, arr is not empty
    arr = np.array([1, 2, 3, 0, 1, 2]).astype(np.int32)
    size = np.array(0).astype(np.int32)
    weights = np.array([]).astype(np.float64)  #weights needs to be empty
    input_dict = {"name": "bincount_7", "arr": arr, "size": size, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different size and arr values
    arr = np.array([0, 0, 1, 1, 2, 2, 2]).astype(np.int32)
    size = np.array(3).astype(np.int32)
    weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]).astype(np.float32)
    input_dict = {"name": "bincount_8", "arr": arr, "size": size, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Int weights with different values.
    arr = np.array([0, 0, 1, 1, 2, 2, 2]).astype(np.int32)
    size = np.array(4).astype(np.int32)
    weights = np.array([1, 0, 1, 0, 1, 0, 1]).astype(np.int32)
    input_dict = {"name": "bincount_9", "arr": arr, "size": size, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: larger size and arr values.
    arr = np.array([1, 5, 2, 3, 4, 1, 0, 5]).astype(np.int32)
    size = np.array(7).astype(np.int32)
    weights = np.array([0.1, 0.5, 0.2, 0.3, 0.4, 0.1, 0.0, 0.5]).astype(np.float32)
    input_dict = {"name": "bincount_10", "arr": arr, "size": size, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Bincount"] = tf_raw_ops_bincount_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Bincount' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Bincount'.")

check_valid('tf.raw_ops.Bincount', generated_inputs['tf.raw_ops.Bincount'], lib="tf", suffix=0)
