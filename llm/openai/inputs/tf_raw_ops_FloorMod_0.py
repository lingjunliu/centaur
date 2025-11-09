
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_floormod_inputs():
    list_of_inputs = []

    x = np.array(7, dtype=np.int32)
    y = np.array(3, dtype=np.int32)
    input_dict = {"name": "case_int32_scalar_pos", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-7, dtype=np.int32)
    y = np.array(3, dtype=np.int32)
    input_dict = {"name": "case_int32_scalar_neg", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-7, -1, 0, 1, 7], dtype=np.int32)
    y = np.array([3, -3, 2, -2, 5], dtype=np.int32)
    input_dict = {"name": "case_int32_vector_mixed", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[10, -10], [5, -5]], dtype=np.int64)
    y = np.array(6, dtype=np.int64)
    input_dict = {"name": "case_int64_matrix_scalar_broadcast", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([255, 128, 0], dtype=np.uint8)
    y = np.array([10, 7, 13], dtype=np.uint8)
    input_dict = {"name": "case_uint8_vector", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1000, -2000, 3000], [4000, -5000, 6000]], dtype=np.int16)
    y = np.array(-4, dtype=np.int16)
    input_dict = {"name": "case_int16_matrix_scalar_neg_divisor", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-7.5, 7.5, 0.0, -0.1], dtype=np.float32)
    y = np.array([3.0, -3.0, 2.0, 0.2], dtype=np.float32)
    input_dict = {"name": "case_float32_vector_mixed", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-5.0], [0.5], [8.0]], dtype=np.float64)
    y = np.array([[2.0, -2.0, 3.0, -3.0]], dtype=np.float64)
    input_dict = {"name": "case_float64_broadcast_2d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[ -1], [ -8], [  7]], [[  4], [-12], [ 15]]], dtype=np.int8)
    y = np.array([3, -5, 10], dtype=np.int8)
    input_dict = {"name": "case_int8_broadcast_3d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    y = np.array(-1.5, dtype=np.float32)
    input_dict = {"name": "case_float32_vector_scalar_neg_divisor", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = (np.arange(24, dtype=np.int32).reshape(1, 2, 3, 4) - 12).astype(np.int32)
    y = np.array([2, -2, 5, -7], dtype=np.int32)
    input_dict = {"name": "case_int32_4d_broadcast_lastdim", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.int64(2**62), np.int64(-2**62), np.int64(2**61 + 7)], dtype=np.int64)
    y = np.array(97, dtype=np.int64)
    input_dict = {"name": "case_int64_large_values_scalar_divisor", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FloorMod"] = tf_raw_ops_floormod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.FloorMod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FloorMod'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.FloorMod', generated_inputs['tf.raw_ops.FloorMod'], lib="tf", suffix=0)
