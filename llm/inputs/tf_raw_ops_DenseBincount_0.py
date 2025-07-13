
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DenseBincount_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([0, 1, 1, 2, 4, 3, 2, 1, 7]).astype(np.int32)
    size_tensor = np.array(8).astype(np.int32)
    weights_tensor = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.float32)
    binary_output_bool = False
    name_str = "bincount_1"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "weights": weights_tensor,
        "binary_output": binary_output_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([0, 1, 2, 3, 4]).astype(np.int64)
    size_tensor = np.array(6).astype(np.int64)
    weights_tensor = np.array([1, 1, 1, 1, 1]).astype(np.int64)
    binary_output_bool = True
    name_str = "bincount_2"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "weights": weights_tensor,
        "binary_output": binary_output_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([0, 0, 1, 1, 2, 2, 2]).astype(np.int32)
    size_tensor = np.array(3).astype(np.int32)
    weights_tensor = np.array([]).astype(np.float32)
    binary_output_bool = False
    name_str = "bincount_3"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "weights": weights_tensor,
        "binary_output": binary_output_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([5, 4, 3, 2, 1, 0]).astype(np.int64)
    size_tensor = np.array(7).astype(np.int64)
    weights_tensor = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).astype(np.float64)
    binary_output_bool = True
    name_str = "bincount_4"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "weights": weights_tensor,
        "binary_output": binary_output_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]).astype(np.int32)
    size_tensor = np.array(6).astype(np.int32)
    weights_tensor = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).astype(np.int32)
    binary_output_bool = False
    name_str = "bincount_5"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "weights": weights_tensor,
        "binary_output": binary_output_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    input_tensor = np.array([[0, 1, 2], [3, 4, 0]]).astype(np.int32)
    size_tensor = np.array(5).astype(np.int32)
    weights_tensor = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
    binary_output_bool = False
    name_str = "bincount_6"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "weights": weights_tensor,
        "binary_output": binary_output_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[0, 1], [1, 0]]).astype(np.int64)
    size_tensor = np.array(3).astype(np.int64)
    weights_tensor = np.array([1, 1, 1, 1]).astype(np.int64)
    binary_output_bool = True
    name_str = "bincount_7"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "weights": weights_tensor,
        "binary_output": binary_output_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([0, 1, 2, 3, 4, 5]).astype(np.int32)
    size_tensor = np.array(10).astype(np.int32)
    weights_tensor = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).astype(np.float64)
    binary_output_bool = False
    name_str = "bincount_8"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "weights": weights_tensor,
        "binary_output": binary_output_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([0, 1, 2, 3]).astype(np.int64)
    size_tensor = np.array(4).astype(np.int64)
    weights_tensor = np.array([1, 2, 3, 4]).astype(np.int32)
    binary_output_bool = False
    name_str = "bincount_9"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "weights": weights_tensor,
        "binary_output": binary_output_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([0, 1, 2]).astype(np.int32)
    size_tensor = np.array(5).astype(np.int32)
    weights_tensor = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    binary_output_bool = True
    name_str = "bincount_10"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "weights": weights_tensor,
        "binary_output": binary_output_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
dense_bincount_inputs = tf_raw_ops_DenseBincount_inputs()
generated_inputs["tf.raw_ops.DenseBincount"] = dense_bincount_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DenseBincount' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DenseBincount'.")

check_valid('tf.raw_ops.DenseBincount', generated_inputs['tf.raw_ops.DenseBincount'], lib="tf", suffix=0)
