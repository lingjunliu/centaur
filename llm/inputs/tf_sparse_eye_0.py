
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_eye_inputs():
    list_of_inputs = []

    # Input 1
    num_rows = 5
    num_columns = 5
    dtype = tf.float32
    name = "eye1"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    num_rows = 3
    num_columns = 7
    dtype = tf.float64
    name = "eye2"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    num_rows = 10
    num_columns = 10
    dtype = tf.int32
    name = "eye3"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    num_rows = 2
    num_columns = 2
    dtype = tf.complex64
    name = "eye4"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    num_rows = 8
    num_columns = 4
    dtype = tf.bfloat16
    name = "eye5"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    num_rows = 6
    num_columns = 6
    dtype = tf.float16
    name = "eye6"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    num_rows = 9
    num_columns = 9
    dtype = tf.int64
    name = "eye7"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    num_rows = 4
    num_columns = 8
    dtype = tf.uint8
    name = "eye8"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    num_rows = 7
    num_columns = 3
    dtype = tf.int8
    name = "eye9"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    num_rows = 1
    num_columns = 1
    dtype = tf.bool
    name = "eye10"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.eye"] = tf_sparse_eye_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.eye' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.eye'.")

check_valid('tf.sparse.eye', generated_inputs['tf.sparse.eye'], lib="tf", suffix=0)
