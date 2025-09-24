
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_empty_inputs():
    list_of_inputs = []

    shape = np.array([2, 3], dtype=np.int32)
    dtype = np.float32
    init = False
    name = "empty_f32_2x3_noinit"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([5], dtype=np.int32)
    dtype = np.int64
    init = True
    name = "empty_i64_5_init"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([], dtype=np.int32)
    dtype = np.bool_
    init = True
    name = "empty_bool_scalar_init"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([0], dtype=np.int32)
    dtype = np.float64
    init = False
    name = "empty_f64_len0_noinit"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([3, 0, 4], dtype=np.int32)
    dtype = np.int32
    init = True
    name = "empty_i32_3x0x4_init"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([1, 1, 1, 1], dtype=np.int32)
    dtype = np.complex64
    init = True
    name = "empty_c64_1x1x1x1_init"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([10, 10], dtype=np.int32)
    dtype = np.uint8
    init = False
    name = "empty_u8_10x10_noinit"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([2, 3, 4], dtype=np.int32)
    dtype = np.float16
    init = True
    name = "empty_f16_2x3x4_init"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([7], dtype=np.int32)
    dtype = np.complex128
    init = False
    name = "empty_c128_len7_noinit"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([0, 0], dtype=np.int32)
    dtype = np.int8
    init = True
    name = "empty_i8_0x0_init"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([2], dtype=np.int32)
    dtype = np.uint16
    init = False
    name = "empty_u16_len2_noinit"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([4, 1, 2, 0], dtype=np.int32)
    dtype = np.float32
    init = True
    name = "empty_f32_4x1x2x0_init"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Empty"] = tf_raw_ops_empty_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Empty' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Empty'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Empty', generated_inputs['tf.raw_ops.Empty'], lib="tf", suffix=0)
