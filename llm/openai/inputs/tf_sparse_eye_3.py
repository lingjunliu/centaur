
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_eye_inputs():
    list_of_inputs = []

    # Input 1
    num_rows = np.array(5, dtype=np.int32)
    num_columns = 5
    dtype = np.float32
    name = "eye_square_f32"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    num_rows = np.array(0, dtype=np.int32)
    num_columns = 0
    dtype = np.float64
    name = "eye_empty"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    num_rows = np.array(7, dtype=np.int64)
    num_columns = 3
    dtype = np.int32
    name = "eye_tall_int32"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    num_rows = np.array(3, dtype=np.int64)
    num_columns = 7
    dtype = np.int64
    name = "eye_wide_int64"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    num_rows = np.array(1, dtype=np.int32)
    num_columns = 1
    dtype = np.bool_
    name = "eye_bool_1x1"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    num_rows = np.array(4, dtype=np.int64)
    num_columns = 4
    dtype = np.complex64
    name = "eye_c64_4x4"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    num_rows = np.array(2, dtype=np.int32)
    num_columns = 5
    dtype = np.complex128
    name = "eye_c128_2x5"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    num_rows = np.array(10, dtype=np.int64)
    num_columns = 10
    dtype = np.float16
    name = "eye_f16_10x10"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    num_rows = np.array(6, dtype=np.int32)
    num_columns = 4
    dtype = np.uint8
    name = "eye_uint8_6x4"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    num_rows = np.array(8, dtype=np.int64)
    num_columns = 8
    dtype = np.float64
    name = "eye_double"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    num_rows = np.array(12, dtype=np.int64)
    num_columns = 0
    dtype = np.int16
    name = "eye_zero_cols"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    num_rows = np.array(0, dtype=np.int32)
    num_columns = 5
    dtype = np.float32
    name = "eye_zero_rows"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sparse.eye_3"] = tf_sparse_eye_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.eye_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.eye_3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.eye', generated_inputs['tf.sparse.eye_3'], lib="tf", suffix=3)
