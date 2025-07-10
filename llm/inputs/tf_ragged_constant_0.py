
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_constant_inputs():
    list_of_inputs = []

    # Input 1
    pylist = [[1, 2], [3], [4, 5, 6]]
    dtype = np.int32
    ragged_rank = 1
    inner_shape = ()
    name = "ragged_tensor_1"
    row_splits_dtype = tf.int64

    input_dict = {
        "pylist": pylist,
        "dtype": dtype,
        "ragged_rank": ragged_rank,
        "inner_shape": inner_shape,
        "name": name,
        "row_splits_dtype": row_splits_dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    pylist = [[[1, 2], [3]], [[4, 5, 6]]]
    dtype = np.float32
    ragged_rank = 2
    inner_shape = ()
    name = "ragged_tensor_2"
    row_splits_dtype = tf.int32

    input_dict = {
        "pylist": pylist,
        "dtype": dtype,
        "ragged_rank": ragged_rank,
        "inner_shape": inner_shape,
        "name": name,
        "row_splits_dtype": row_splits_dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    pylist = [[], [1, 2], []]
    dtype = np.int64
    ragged_rank = 1
    inner_shape = ()
    name = "ragged_tensor_3"
    row_splits_dtype = tf.int64

    input_dict = {
        "pylist": pylist,
        "dtype": dtype,
        "ragged_rank": ragged_rank,
        "inner_shape": inner_shape,
        "name": name,
        "row_splits_dtype": row_splits_dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    pylist = [[1], [2, 3, 4], [5, 6]]
    dtype = np.float64
    ragged_rank = 1
    inner_shape = ()
    name = "ragged_tensor_4"
    row_splits_dtype = tf.int32

    input_dict = {
        "pylist": pylist,
        "dtype": dtype,
        "ragged_rank": ragged_rank,
        "inner_shape": inner_shape,
        "name": name,
        "row_splits_dtype": row_splits_dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.ragged.constant"] = tf_ragged_constant_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.ragged.constant' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.constant'.")

check_valid('tf.ragged.constant', generated_inputs['tf.ragged.constant'], lib="tf", suffix=0)
