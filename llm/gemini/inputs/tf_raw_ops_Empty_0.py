
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_empty_inputs():
    list_of_inputs = []

    # Input 1
    shape = np.array([2, 3], dtype=np.int32)
    dtype = tf.float32
    init = False
    name = "empty_tensor_1"
    input_dict = {"shape": shape, "dtype": dtype, "init": init, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = np.array([5], dtype=np.int32)
    dtype = tf.int32
    init = True
    name = "empty_tensor_2"
    input_dict = {"shape": shape, "dtype": dtype, "init": init, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = np.array([1, 2, 3, 4], dtype=np.int32)
    dtype = tf.bool
    init = False
    name = "empty_tensor_3"
    input_dict = {"shape": shape, "dtype": dtype, "init": init, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = np.array([0], dtype=np.int32)
    dtype = tf.float64
    init = True
    name = "empty_tensor_4"
    input_dict = {"shape": shape, "dtype": dtype, "init": init, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = np.array([10, 10], dtype=np.int32)
    dtype = tf.complex64
    init = False
    name = "empty_tensor_5"
    input_dict = {"shape": shape, "dtype": dtype, "init": init, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = np.array([1], dtype=np.int32)
    dtype = tf.string
    init = True
    name = "empty_tensor_6"
    input_dict = {"shape": shape, "dtype": dtype, "init": init, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = np.array([2, 2, 2], dtype=np.int32)
    dtype = tf.uint8
    init = False
    name = "empty_tensor_7"
    input_dict = {"shape": shape, "dtype": dtype, "init": init, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = np.array([1, 5], dtype=np.int32)
    dtype = tf.int64
    init = True
    name = "empty_tensor_8"
    input_dict = {"shape": shape, "dtype": dtype, "init": init, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = np.array([3, 1, 4, 1], dtype=np.int32)
    dtype = tf.bfloat16
    init = False
    name = "empty_tensor_9"
    input_dict = {"shape": shape, "dtype": dtype, "init": init, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = np.array([1, 1], dtype=np.int32)
    dtype = tf.float16
    init = False
    name = "empty_tensor_10"
    input_dict = {"shape": shape, "dtype": dtype, "init": init, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Empty"] = tf_raw_ops_empty_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Empty' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Empty'.")

check_valid('tf.raw_ops.Empty', generated_inputs['tf.raw_ops.Empty'], lib="tf", suffix=0)
