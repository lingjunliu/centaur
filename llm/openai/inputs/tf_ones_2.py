
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ones_2_inputs():
    list_of_inputs = []

    shape = (3, 4)
    dtype = np.int32
    name = "ones_i32_3x4"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (0,)
    dtype = np.float32
    name = "ones_f32_len0"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = ()
    dtype = np.float64
    name = "ones_scalar_f64"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (2, 0, 3)
    dtype = np.bool_
    name = "ones_bool_2x0x3"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (1,)
    dtype = np.complex64
    name = "ones_c64_len1"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (5, 1)
    dtype = np.uint8
    name = "ones_u8_5x1"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (2, 3, 4, 5)
    dtype = np.dtype('float16')
    name = "ones_f16_2x3x4x5"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (1, 2, 3)
    dtype = np.int64
    name = "ones_i64_1x2x3"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (10,)
    dtype = np.dtype('complex128')
    name = "ones_c128_len10"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (2, 2, 2, 2, 2)
    dtype = np.int16
    name = "ones_i16_5d"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (1, 0, 0)
    dtype = np.dtype('uint16')
    name = "ones_u16_1x0x0"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (7,)
    dtype = np.float32
    name = "ones_f32_len7_unicode_名"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    return list_of_inputs

generated_inputs["tf.ones_2"] = tf_ones_2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.ones_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ones_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.ones', generated_inputs['tf.ones_2'], lib="tf", suffix=2)
