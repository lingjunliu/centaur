
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_eye_inputs():
    list_of_inputs = []

    # Input 1
    num_rows = np.int32(2)
    num_columns = np.int32(2)
    batch_shape = []
    dtype = tf.float32
    name = "eye1"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "batch_shape": batch_shape,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    num_rows = np.int32(3)
    num_columns = np.int32(4)
    batch_shape = [np.int32(2)]
    dtype = tf.float64
    name = "eye2"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "batch_shape": batch_shape,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    num_rows = np.int32(5)
    num_columns = np.int32(5)
    batch_shape = [np.int32(2), np.int32(3)]
    dtype = tf.float16
    name = "eye3"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "batch_shape": batch_shape,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    num_rows = np.int32(1)
    num_columns = np.int32(1)
    batch_shape = [np.int32(4), np.int32(2), np.int32(1)]
    dtype = tf.int32
    name = "eye4"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "batch_shape": batch_shape,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    num_rows = np.int32(4)
    num_columns = None
    batch_shape = []
    dtype = tf.complex64
    name = "eye5"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "batch_shape": batch_shape,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    num_rows = np.int32(6)
    num_columns = np.int32(7)
    batch_shape = [np.int32(1)]
    dtype = tf.bfloat16
    name = "eye6"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "batch_shape": batch_shape,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    num_rows = np.int32(2)
    num_columns = np.int32(3)
    batch_shape = [np.int32(5)]
    dtype = tf.complex128
    name = "eye7"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "batch_shape": batch_shape,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    num_rows = np.int32(8)
    num_columns = np.int32(8)
    batch_shape = [np.int32(2), np.int32(2)]
    dtype = tf.float32
    name = "eye8"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "batch_shape": batch_shape,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    num_rows = np.int32(10)
    num_columns = None
    batch_shape = [np.int32(2), np.int32(3), np.int32(4)]
    dtype = tf.float32
    name = "eye9"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "batch_shape": batch_shape,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    num_rows = np.int32(7)
    num_columns = np.int32(9)
    batch_shape = [np.int32(3)]
    dtype = tf.float32
    name = "eye10"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "batch_shape": batch_shape,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.eye"] = tf_eye_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.eye' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.eye'.")

check_valid('tf.eye', generated_inputs['tf.eye'], lib="tf", suffix=0)
