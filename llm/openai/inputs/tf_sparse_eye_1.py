
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_eye_inputs():
    list_of_inputs = []

    # Input 1
    num_rows = np.int32(0)
    num_columns = np.int32(0)
    dtype = np.float32
    name = "eye_zero_zero"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    num_rows = np.int32(1)
    num_columns = np.int32(1)
    dtype = np.int32
    name = "eye_one"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    num_rows = np.int64(3)
    num_columns = np.int64(5)
    dtype = np.float64
    name = "eye_rect64"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    num_rows = np.int32(5)
    num_columns = np.int32(3)
    dtype = np.float32
    name = "eye_rect"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    num_rows = np.int64(10)
    num_columns = np.int64(10)
    dtype = np.int64
    name = "eye_int64"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    num_rows = np.int32(7)
    num_columns = np.int32(7)
    dtype = np.complex64
    name = "eye_c64"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    num_rows = np.int32(4)
    num_columns = np.int32(4)
    dtype = np.complex128
    name = "eye_c128"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    num_rows = np.int32(2)
    num_columns = np.int32(8)
    dtype = np.bool_
    name = "eye_bool"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    num_rows = np.int32(9)
    num_columns = np.int32(9)
    dtype = np.float16
    name = "eye_f16"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    num_rows = np.int64(12)
    num_columns = np.int64(15)
    dtype = np.float32
    name = "eye_12_15"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    num_rows = np.int32(100)
    num_columns = np.int32(100)
    dtype = np.float32
    name = "eye_100"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    num_rows = np.int32(3)
    num_columns = np.int32(3)
    dtype = np.uint8
    name = "eye_uint8"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sparse.eye_1"] = tf_sparse_eye_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.eye_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.eye_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.eye', generated_inputs['tf.sparse.eye_1'], lib="tf", suffix=1)
