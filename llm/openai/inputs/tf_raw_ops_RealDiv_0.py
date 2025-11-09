
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_realdiv_inputs():
    list_of_inputs = []

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    input_dict = {"name": "case_float32_vector", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, -2.0], [3.5, -4.5]], dtype=np.float64)
    y = np.array(2.0, dtype=np.float64)
    input_dict = {"name": "case_scalar_divisor_float64", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[-5.0, -6.0], [7.0, 8.0]]], dtype=np.float16)
    y = np.array([[[1.0, -1.0], [2.0, -2.0]], [[-1.0, 1.0], [0.5, -0.5]]], dtype=np.float16)
    input_dict = {"name": "case_float16_3d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([[1.0, -1.0, 2.0, -2.0]], dtype=np.float32)
    input_dict = {"name": "case_broadcast_2d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1 + 2j, -3 + 4j, -1 - 1j], dtype=np.complex64)
    y = np.array([1 - 1j, 2 + 0j, -0.5 + 0.5j], dtype=np.complex64)
    input_dict = {"name": "case_complex64_vector", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1 + 0j, 0 + 1j, -1 - 1j], [2 - 2j, -3 + 0j, 4 + 4j]], dtype=np.complex128)
    y = np.array([1 + 0j, -1 + 1j, 2 - 1j], dtype=np.complex128)
    input_dict = {"name": "case_complex128_broadcast", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, -2.0, 0.0], dtype=np.float32)
    y = np.array([0.0, 2.0, -0.0], dtype=np.float32)
    input_dict = {"name": "case_zero_division", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = (np.arange(6, dtype=np.float64).reshape(2, 1, 3, 1) - 2.5)
    y = np.array([[1.0], [-2.0], [3.0]], dtype=np.float64)
    input_dict = {"name": "case_float64_4d_broadcast", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(10.0, dtype=np.float32)
    y = np.array([1.0, -2.0, 5.0, -10.0], dtype=np.float32)
    input_dict = {"name": "case_scalar_numerator", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1000.0, 0.125, 65504.0, -0.0005, 1.5], dtype=np.float16)
    y = np.array([2.0, -0.5, 256.0, 2.0, -3.0], dtype=np.float16)
    input_dict = {"name": "case_float16_edge_values", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(3 + 4j, dtype=np.complex128)
    y = np.array([[1 - 1j, 2 + 2j], [-3 + 0j, 4 - 4j]], dtype=np.complex128)
    input_dict = {"name": "case_complex128_scalar_over_matrix", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[np.inf, -np.inf], [np.nan, 1.0]], dtype=np.float32)
    y = np.array([[1.0, -2.0], [3.0, np.nan]], dtype=np.float32)
    input_dict = {"name": "case_nan_inf", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.RealDiv"] = tf_raw_ops_realdiv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RealDiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RealDiv'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.RealDiv', generated_inputs['tf.raw_ops.RealDiv'], lib="tf", suffix=0)
