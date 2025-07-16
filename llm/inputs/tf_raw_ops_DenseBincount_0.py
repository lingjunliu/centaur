
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_dense_bincount_inputs():
    list_of_inputs = []

    # Input 1: Basic case with int32 and weights
    input_arr = np.array([1, 2, 3, 0, 1, 2]).astype(np.int32)
    size_val = np.array(5).astype(np.int32)
    weights_arr = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]).astype(np.float32)
    binary_output_val = False
    name_val = None

    input_dict = {
        "input": input_arr,
        "size": size_val,
        "weights": weights_arr,
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2:  int64 input, int64 weights, binary output
    input_arr = np.array([5, 2, 1, 0, 3, 2, 5]).astype(np.int64)
    size_val = np.array(6).astype(np.int64)
    weights_arr = np.array([1, 1, 1, 1, 1, 1, 1]).astype(np.int64)
    binary_output_val = True
    name_val = None

    input_dict = {
        "input": input_arr,
        "size": size_val,
        "weights": weights_arr,
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty weights, int32 input
    input_arr = np.array([0, 1, 2, 3, 4, 0, 1]).astype(np.int32)
    size_val = np.array(5).astype(np.int32)
    weights_arr = np.array([]).astype(np.float32)
    binary_output_val = False
    name_val = None

    input_dict = {
        "input": input_arr,
        "size": size_val,
        "weights": weights_arr,
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float64 weights
    input_arr = np.array([0, 1, 2, 0, 1, 2]).astype(np.int32)
    size_val = np.array(3).astype(np.int32)
    weights_arr = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]).astype(np.float64)
    binary_output_val = False
    name_val = None

    input_dict = {
        "input": input_arr,
        "size": size_val,
        "weights": weights_arr,
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  int64 weights and binary output
    input_arr = np.array([0, 1, 2, 0, 1, 2]).astype(np.int32)
    size_val = np.array(3).astype(np.int32)
    weights_arr = np.array([1, 2, 3, 4, 5, 6]).astype(np.int64)
    binary_output_val = True
    name_val = None

    input_dict = {
        "input": input_arr,
        "size": size_val,
        "weights": weights_arr,
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: size 1
    input_arr = np.array([0]).astype(np.int32)
    size_val = np.array(1).astype(np.int32)
    weights_arr = np.array([1.0]).astype(np.float32)
    binary_output_val = False
    name_val = None

    input_dict = {
        "input": input_arr,
        "size": size_val,
        "weights": weights_arr,
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  int32 weights
    input_arr = np.array([0, 1, 2, 0, 1, 2]).astype(np.int32)
    size_val = np.array(4).astype(np.int32)
    weights_arr = np.array([1, 2, 3, 4, 5, 6]).astype(np.int32)
    binary_output_val = False
    name_val = None

    input_dict = {
        "input": input_arr,
        "size": size_val,
        "weights": weights_arr,
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D input
    input_arr = np.array([[0, 1], [2, 0]]).astype(np.int32)
    size_val = np.array(3).astype(np.int32)
    weights_arr = np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.float32)
    binary_output_val = False
    name_val = None

    input_dict = {
        "input": input_arr,
        "size": size_val,
        "weights": weights_arr,
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int64 input and size
    input_arr = np.array([1, 2, 3, 0, 1, 2]).astype(np.int64)
    size_val = np.array(5).astype(np.int64)
    weights_arr = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]).astype(np.float32)
    binary_output_val = False
    name_val = None

    input_dict = {
        "input": input_arr,
        "size": size_val,
        "weights": weights_arr,
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: binary_output = True and zero size
    input_arr = np.array([0, 0, 0]).astype(np.int32)
    size_val = np.array(1).astype(np.int32)
    weights_arr = np.array([1,1,1]).astype(np.int32)
    binary_output_val = True
    name_val = None

    input_dict = {
        "input": input_arr,
        "size": size_val,
        "weights": weights_arr,
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DenseBincount"] = tf_raw_ops_dense_bincount_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DenseBincount' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DenseBincount'.")

check_valid('tf.raw_ops.DenseBincount', generated_inputs['tf.raw_ops.DenseBincount'], lib="tf", suffix=0)
