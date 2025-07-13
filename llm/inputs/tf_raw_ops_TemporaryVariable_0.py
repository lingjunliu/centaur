
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_TemporaryVariable_inputs():
    list_of_inputs = []

    # Input 1
    shape = [2, 3]
    dtype = tf.float32
    var_name = "temp_var_1"
    name = "op_1"
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = [5]
    dtype = tf.int32
    var_name = ""
    name = None
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = [1, 4, 2]
    dtype = tf.complex64
    var_name = "complex_var"
    name = "complex_op"
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = [10, 10]
    dtype = tf.bool
    var_name = "bool_var"
    name = "bool_op"
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = [2, 2, 2, 2]
    dtype = tf.float64
    var_name = "float64_var"
    name = "float64_op"
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = [7]
    dtype = tf.int64
    var_name = "int64_var"
    name = "int64_op"
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = []
    dtype = tf.string
    var_name = "string_var"
    name = "string_op"
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = [1]
    dtype = tf.uint8
    var_name = "uint8_var"
    name = "uint8_op"
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = [3, 1]
    dtype = tf.bfloat16
    var_name = "bfloat16_var"
    name = "bfloat16_op"
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = [4, 4, 4]
    dtype = tf.float16
    var_name = "float16_var"
    name = "float16_op"
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.TemporaryVariable"] = tf_raw_ops_TemporaryVariable_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.TemporaryVariable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.TemporaryVariable'.")

check_valid('tf.raw_ops.TemporaryVariable', generated_inputs['tf.raw_ops.TemporaryVariable'], lib="tf", suffix=0)
