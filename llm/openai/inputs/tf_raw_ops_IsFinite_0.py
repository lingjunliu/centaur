
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_isfinite_inputs():
    list_of_inputs = []

    x = np.array([5.0, 4.8, 6.8, np.inf, np.nan], dtype=np.float32)
    input_dict = {"name": "finite_vector_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0.0, -1.0, 2.5], [np.inf, -np.inf, np.nan]], dtype=np.float64)
    input_dict = {"name": "matrix_mixed_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0, -2.0], [3.0, -4.0]], [[np.nan, np.inf], [-np.inf, 0.0]]], dtype=np.float16)
    input_dict = {"name": "tensor3d_f16", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(0.0, dtype=np.float32)
    input_dict = {"name": "scalar_zero_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(np.nan, dtype=np.float64)
    input_dict = {"name": "scalar_nan_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    input_dict = {"name": "empty_1d_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.empty((2, 0), dtype=np.float64)
    input_dict = {"name": "empty_2d_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[[1.0, np.inf, -np.inf]], [[-0.0, 3.14, np.nan]]]], dtype=np.float32)
    input_dict = {"name": "tensor4d_mixed_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e308, -1e308, np.inf, -np.inf, np.nan], dtype=np.float64)
    input_dict = {"name": "large_values_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-45, -1e-45, 1.1754944e-38, -1.1754944e-38], dtype=np.float32)
    input_dict = {"name": "subnormal_normal_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0, 65504.0, 1e5, -1e5, np.nan], dtype=np.float16)
    input_dict = {"name": "f16_extremes", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[-np.inf], [np.nan]], [[1.0], [2.0]]], dtype=np.float32)
    input_dict = {"name": "mixed_column_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.IsFinite"] = tf_raw_ops_isfinite_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.IsFinite' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.IsFinite'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.IsFinite', generated_inputs['tf.raw_ops.IsFinite'], lib="tf", suffix=0)
