
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ones_inputs():
    list_of_inputs = []

    # 1
    shape = np.array([3, 4], dtype=np.int32)
    dtype = np.int32
    name = "ones_int32_2d"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 2
    shape = np.array([2, 0], dtype=np.int32)
    dtype = np.float32
    name = "ones_float32_with_zero_dim"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 3
    shape = np.array([5], dtype=np.int32)
    dtype = np.bool_
    name = "ones_bool_1d"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 4
    shape = np.array([1, 1, 1], dtype=np.int32)
    dtype = np.float64
    name = "ones_float64_3d"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 5
    shape = np.array([], dtype=np.int32)
    dtype = np.float32
    name = "ones_scalar_default_float32"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 6
    shape = np.array([2, 3, 4], dtype=np.int32)
    dtype = np.complex64
    name = "ones_complex64_3d"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 7
    shape = np.array([1, 2, 3, 4], dtype=np.int32)
    dtype = np.int64
    name = "ones_int64_4d"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 8
    shape = np.array([10], dtype=np.int32)
    dtype = np.uint8
    name = "ones_uint8_1d_len10"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 9
    shape = np.array([2, 2], dtype=np.int32)
    dtype = np.float16
    name = "ones_float16_2x2"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 10
    shape = np.array([0, 3, 0], dtype=np.int32)
    dtype = np.float16
    name = "ones_float16_with_zeros"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 11
    shape = np.array([4, 1], dtype=np.int32)
    dtype = np.complex128
    name = "ones_complex128_2d"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 12
    shape = np.array([1], dtype=np.int32)
    dtype = np.int8
    name = "ones_int8_singleton"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    return list_of_inputs

generated_inputs["tf.ones_3"] = tf_ones_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.ones_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ones_3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.ones', generated_inputs['tf.ones_3'], lib="tf", suffix=3)
