
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_variable_inputs():
    list_of_inputs = []

    # Input 1
    shape = [2, 3]
    dtype = tf.float32
    container = ""
    shared_name = ""
    name = "var1"
    input_dict = {"shape": shape, "dtype": dtype, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = [1, 5, 7]
    dtype = tf.int32
    container = "my_container"
    shared_name = "shared_var"
    name = "var2"
    input_dict = {"shape": shape, "dtype": dtype, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = [10]
    dtype = tf.bool
    container = ""
    shared_name = "bool_var"
    name = "var3"
    input_dict = {"shape": shape, "dtype": dtype, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = [2, 2, 2, 2]
    dtype = tf.complex64
    container = "complex_container"
    shared_name = ""
    name = "var4"
    input_dict = {"shape": shape, "dtype": dtype, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = []  # Scalar
    dtype = tf.string
    container = ""
    shared_name = "scalar_string"
    name = "var5"
    input_dict = {"shape": shape, "dtype": dtype, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = [3, 4, 5]
    dtype = tf.float64
    container = "double_container"
    shared_name = "double_var"
    name = "var6"
    input_dict = {"shape": shape, "dtype": dtype, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = [7, 1]
    dtype = tf.int16
    container = ""
    shared_name = "int16_var"
    name = "var7"
    input_dict = {"shape": shape, "dtype": dtype, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = [1, 1, 1, 1, 1]
    dtype = tf.uint8
    container = "uint8_container"
    shared_name = ""
    name = "var8"
    input_dict = {"shape": shape, "dtype": dtype, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = [4]
    dtype = tf.qint8
    container = ""
    shared_name = "qint8_var"
    name = "var9"
    input_dict = {"shape": shape, "dtype": dtype, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = [8, 8]
    dtype = tf.quint8
    container = "quint8_container"
    shared_name = "my_quint8"
    name = "var10"
    input_dict = {"shape": shape, "dtype": dtype, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Variable"] = tf_raw_ops_variable_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Variable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Variable'.")

check_valid('tf.raw_ops.Variable', generated_inputs['tf.raw_ops.Variable'], lib="tf", suffix=0)
