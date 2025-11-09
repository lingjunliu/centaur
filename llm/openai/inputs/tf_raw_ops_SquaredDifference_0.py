
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

np.random.seed(42)

def tf_raw_ops_squared_difference_inputs():
    list_of_inputs = []

    x = np.array([1.0, -2.5, 3.3], dtype=np.float32)
    y = np.array([0.5, 2.0, -3.3], dtype=np.float32)
    input_dict = {"name": "case1_f32_1d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.int32)
    y = np.array([[0, 2, -3], [5, -7, 9]], dtype=np.int32)
    input_dict = {"name": "case2_i32_2d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(3.14, dtype=np.float64)
    y = np.array([[[1.0, -2.0, 0.0], [4.0, -5.5, 6.6]]], dtype=np.float64)
    input_dict = {"name": "case3_f64_scalar_broadcast", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, -1.5, 2.0, -2.5],
                  [3.0, -3.5, 4.0, -4.5],
                  [5.0, -5.5, 6.0, -6.5]], dtype=np.float16)
    y = np.array([0.5, -0.5, 1.5, -1.5], dtype=np.float16)
    input_dict = {"name": "case4_f16_broadcast_2d_1d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1]], [[-2]]], dtype=np.int64)
    y = np.array([[[3, -4, 5, -6],
                   [7, -8, 9, -10],
                   [11, -12, 13, -14]]], dtype=np.int64)
    input_dict = {"name": "case5_i64_broadcast_3d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1+2j, -3+0.5j, 0-1j, 2-2j], dtype=np.complex64)
    y = np.array(1-1j, dtype=np.complex64)
    input_dict = {"name": "case6_c64_scalar_broadcast", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1+1j, -2-3j], [4-5j, -6+7j]], dtype=np.complex128)
    y = np.array([[0-1j, 2+3j], [-4+5j, 6-7j]], dtype=np.complex128)
    input_dict = {"name": "case7_c128_2d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.empty((0, 3), dtype=np.float32)
    y = np.empty((0, 3), dtype=np.float32)
    input_dict = {"name": "case8_f32_zerosized", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.arange(1*2*3*1, dtype=np.float32).reshape(1, 2, 3, 1)
    y = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    input_dict = {"name": "case9_f32_4d_2d_broadcast", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-7, dtype=np.int32)
    y = np.array(5, dtype=np.int32)
    input_dict = {"name": "case10_i32_scalar", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.inf, -np.inf, np.nan], dtype=np.float64)
    y = np.array([1.0, -2.0, 0.0], dtype=np.float64)
    input_dict = {"name": "case11_f64_nan_inf", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex64)
    y = np.array([1+0j, 0+1j, -1+0j, 0-1j], dtype=np.complex64)
    input_dict = {"name": "case12_c64_3d_1d_broadcast", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SquaredDifference"] = tf_raw_ops_squared_difference_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SquaredDifference' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SquaredDifference'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SquaredDifference', generated_inputs['tf.raw_ops.SquaredDifference'], lib="tf", suffix=0)
