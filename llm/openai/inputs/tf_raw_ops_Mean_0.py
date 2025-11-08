
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_mean_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = np.array([1.0, 2.5, -3.0, 0.0, 4.5], dtype=np.float32)
    axis = np.array(0, dtype=np.int32)
    keep_dims = False
    name = "mean_f32_1d_axis0"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 2
    input_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis = np.array(1, dtype=np.int64)
    keep_dims = True
    name = "mean_i32_2d_axis1_keep"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 3
    input_arr = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    axis = np.array([0, 2], dtype=np.int32)
    keep_dims = False
    name = "mean_u8_3d_axes_0_2"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 4
    input_arr = np.arange(-24, 0, dtype=np.int16).reshape(2, 1, 3, 4)
    axis = np.array(-1, dtype=np.int64)
    keep_dims = True
    name = "mean_i16_4d_axis_-1_keep"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 5
    input_arr = np.array([[-10, 20, -30, 40],
                          [50, -60, 70, -80],
                          [90, -100, 110, -120]], dtype=np.int8)
    axis = np.array([0], dtype=np.int32)
    keep_dims = False
    name = "mean_i8_2d_axis0"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 6
    base = np.arange(6, dtype=np.float32).reshape(3, 2)
    input_arr = (base + 1j * base).astype(np.complex64)
    axis = np.array(0, dtype=np.int32)
    keep_dims = True
    name = "mean_c64_2d_axis0_keep"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 7
    input_arr = (np.ones((2, 2, 2, 2, 2), dtype=np.int64) * 7)
    axis = np.array([1, 3], dtype=np.int64)
    keep_dims = False
    name = "mean_i64_5d_axes_1_3"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 8
    input_arr = np.linspace(-1, 1, num=24, dtype=np.float64).reshape(2, 3, 4)
    axis = np.array([-3], dtype=np.int32)
    keep_dims = True
    name = "mean_f64_3d_axis_-3_keep"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 9
    input_arr = np.array([1 + 2j, -3 + 4j, 5 - 6j], dtype=np.complex128)
    axis = np.array(0, dtype=np.int64)
    keep_dims = False
    name = "mean_c128_1d_axis0"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 10
    input_arr = np.array([[1.5, -2.5],
                          [3.0, 4.0],
                          [-1.0, 0.5]], dtype=np.float16)
    axis = np.array([0, 1], dtype=np.int32)
    keep_dims = False
    name = "mean_f16_2d_all_axes"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 11
    input_arr = np.arange(2*3*4, dtype=np.float32).reshape(2, 3, 4) - 10.0
    axis = np.array([-1, -2], dtype=np.int32)
    keep_dims = True
    name = "mean_f32_3d_axes_-1_-2_keep"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 12
    input_arr = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    axis = np.array([], dtype=np.int32)
    keep_dims = False
    name = "mean_i64_1d_empty_axis"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Mean"] = tf_raw_ops_mean_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Mean' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Mean'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Mean', generated_inputs['tf.raw_ops.Mean'], lib="tf", suffix=0)
