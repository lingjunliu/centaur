
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
    var_name = ""
    name = "temp_var_1"
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = [1, 5, 2]
    dtype = tf.int32
    var_name = "my_temp_var"
    name = None
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = [10]
    dtype = tf.bool
    var_name = ""
    name = "bool_var"
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = [4, 4, 4, 4]
    dtype = tf.int64
    var_name = "very_specific_name"
    name = None
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = [7]
    dtype = tf.complex64
    var_name = ""
    name = "complex_var"
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = [2, 2, 2]
    dtype = tf.float64
    var_name = "double_precision"
    name = None
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = [1]
    dtype = tf.string
    var_name = ""
    name = "string_var"
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = [5, 5]
    dtype = tf.uint8
    var_name = "unsigned_int"
    name = None
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = [3, 1, 3, 1]
    dtype = tf.int8
    var_name = ""
    name = "signed_int"
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    shape = []
    dtype = tf.float16
    var_name = "float16_var"
    name = None
    input_dict = {"shape": shape, "dtype": dtype, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}

try:
    tf.compat.v1.disable_eager_execution()
except Exception as e:
    print(f"Error disabling eager execution: {e}")

generated_inputs["tf.raw_ops.TemporaryVariable"] = tf_raw_ops_TemporaryVariable_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.TemporaryVariable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.TemporaryVariable'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.TemporaryVariable', generated_inputs['tf.raw_ops.TemporaryVariable'], lib="tf", suffix=0)
