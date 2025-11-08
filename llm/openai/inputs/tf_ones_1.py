
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ones_inputs():
    list_of_inputs = []

    shape = [3, 4]
    dtype = np.int32
    name = "ones_int32_2d"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = []
    dtype = np.float32
    name = "ones_scalar_f32"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [0]
    dtype = np.float64
    name = "ones_len0_dim"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [2, 0]
    dtype = np.int64
    name = "ones_zero_second_dim"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [1, 2, 3]
    dtype = np.float16
    name = "ones_3d_f16"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [2, 2, 2, 2]
    dtype = np.complex64
    name = "ones_4d_c64"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [5]
    dtype = np.bool_
    name = "ones_1d_bool"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [2, 3, 4, 5, 6]
    dtype = np.uint8
    name = "ones_5d_u8"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [10]
    dtype = np.int8
    name = "ones_1d_i8"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [2, 3]
    dtype = np.float32
    name = "ones_2d_f32"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [3, 1, 0]
    dtype = np.int16
    name = "ones_zero_last_dim"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [4, 4]
    dtype = np.complex128
    name = "ones_2d_c128"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    return list_of_inputs

generated_inputs["tf.ones_1"] = tf_ones_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.ones_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ones_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.ones', generated_inputs['tf.ones_1'], lib="tf", suffix=1)
