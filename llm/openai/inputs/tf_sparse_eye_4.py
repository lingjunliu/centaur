
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_eye_inputs():
    list_of_inputs = []

    num_rows = np.array(3, dtype=np.int32)
    num_columns = np.array(3, dtype=np.int32)
    dtype = np.float32
    name = "eye_f32_3x3"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(5, dtype=np.int64)
    num_columns = np.array(7, dtype=np.int64)
    dtype = np.float64
    name = "eye_f64_5x7"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(0, dtype=np.int32)
    num_columns = np.array(0, dtype=np.int32)
    dtype = np.float32
    name = "eye_zero_zero"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(4, dtype=np.int64)
    num_columns = np.array(0, dtype=np.int64)
    dtype = np.int32
    name = "eye_i32_4x0"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(0, dtype=np.int32)
    num_columns = np.array(6, dtype=np.int32)
    dtype = np.int64
    name = "eye_i64_0x6"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(1, dtype=np.int64)
    num_columns = np.array(1, dtype=np.int64)
    dtype = np.complex64
    name = "eye_c64_1x1"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(2, dtype=np.int32)
    num_columns = np.array(3, dtype=np.int32)
    dtype = np.float16
    name = "eye_f16_2x3"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(3, dtype=np.int64)
    num_columns = np.array(2, dtype=np.int64)
    dtype = np.complex128
    name = "eye_c128_3x2"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(10, dtype=np.int32)
    num_columns = np.array(10, dtype=np.int32)
    dtype = np.uint8
    name = "eye_u8_10x10"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(8, dtype=np.int64)
    num_columns = np.array(8, dtype=np.int64)
    dtype = np.float32
    name = "éye_unicode_8x8"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(12, dtype=np.int32)
    num_columns = np.array(15, dtype=np.int32)
    dtype = np.float64
    name = "eye_rect_f64_12x15"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(100, dtype=np.int64)
    num_columns = np.array(100, dtype=np.int64)
    dtype = np.float32
    name = "eye_large_100x100"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    return list_of_inputs

generated_inputs["tf.sparse.eye_4"] = tf_sparse_eye_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.eye_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.eye_4'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.eye', generated_inputs['tf.sparse.eye_4'], lib="tf", suffix=4)
