
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sinh_inputs():
    list_of_inputs = []

    x = np.array([-np.inf, -9.0, -0.5, 0.0, 1.0, 2.0, 10.0, np.inf], dtype=np.float32)
    input_dict = {"name": "case_vec_f32_basic", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1.5, 0.0, 1.5], [2.5, -3.3, 4.0]], dtype=np.float64)
    input_dict = {"name": "case_mat_f64_mixed", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(1.2, dtype=np.float16)
    input_dict = {"name": "case_scalar_f16", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.linspace(-2, 2, num=24, dtype=np.float32).reshape(2, 3, 4)
    input_dict = {"name": "case_3d_f32_range", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    input_dict = {"name": "case_empty_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1 + 2j, -1 - 1j, 0 + 0j, 3 - 4j], dtype=np.complex64)
    input_dict = {"name": "case_complex64_vec", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0 + np.pi * 1j, -2.5 + 0j], [1.5 - 1.2j, -0.0 + 0j]], dtype=np.complex128)
    input_dict = {"name": "case_complex128_mat", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.arange(24, dtype=np.float64).reshape(1, 2, 3, 4)
    input_dict = {"name": "case_4d_f64_arange", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-100.0, 100.0, 50.0, -50.0], dtype=np.float32)
    input_dict = {"name": "case_large_magnitude_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.nan, -np.nan, 0.0], dtype=np.float64)
    input_dict = {"name": "case_nan_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0.0, -0.0], [1e-3, -1e-3]], dtype=np.float16)
    input_dict = {"name": "case_f16_small_values", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-0.0, dtype=np.float64)
    input_dict = {"name": "case_scalar_neg_zero_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Sinh"] = tf_raw_ops_sinh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Sinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Sinh'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Sinh', generated_inputs['tf.raw_ops.Sinh'], lib="tf", suffix=0)
