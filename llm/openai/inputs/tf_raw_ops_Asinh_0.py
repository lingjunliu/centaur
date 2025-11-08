
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_asinh_inputs():
    list_of_inputs = []

    x = np.array([-np.inf, -2.0, -0.5, 0.0, 1.0, 1.2, 200.0, 10000.0, np.inf], dtype=np.float32)
    input_dict = {"name": "asinh_f32_1d_range", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-5.0, dtype=np.float64)
    input_dict = {"name": "asinh_f64_scalar_negative", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-10.0, -1.0], [0.0, 1.0], [10.0, 1000.0]], dtype=np.float64)
    input_dict = {"name": "asinh_f64_2d_mixed", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]], dtype=np.float16)
    input_dict = {"name": "asinh_f16_row_vector", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1+0j, 0+1j, -1+2j, -3-4j], dtype=np.complex64)
    input_dict = {"name": "asinh_c64_1d_complex_values", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1-1j, 2+2j], [-3+0.5j, 0-4j], [5+6j, -7-8j]], dtype=np.complex128)
    input_dict = {"name": "asinh_c128_2d_mixed", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([
        [[-np.inf, -3.0, np.nan], [-0.5, 0.0, 0.5]],
        [[1.0, 10.0, 1000.0], [np.inf, -1e-6, 1e-6]]
    ], dtype=np.float32)
    input_dict = {"name": "asinh_f32_3d_with_nan_inf", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.zeros((1, 2, 3, 4), dtype=np.float32)
    input_dict = {"name": "asinh_f32_4d_zeros", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    input_dict = {"name": "asinh_f32_empty_1d", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-12, -1e-12, 1e12, -1e12], dtype=np.float64)
    input_dict = {"name": "asinh_f64_extremes", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([complex(np.inf, 0.0), complex(np.nan, 1.0), complex(1.0, np.inf), complex(-np.inf, -np.inf)], dtype=np.complex64)
    input_dict = {"name": "asinh_c64_with_nan_inf", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.empty((2, 0, 3), dtype=np.float32)
    input_dict = {"name": "asinh_f32_empty_middle_dim", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Asinh"] = tf_raw_ops_asinh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Asinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Asinh'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Asinh', generated_inputs['tf.raw_ops.Asinh'], lib="tf", suffix=0)
