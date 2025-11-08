
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_isinf_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array(np.inf, dtype=np.float32)
    input_dict = {"name": "isinf_scalar_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([5.0, np.inf, 6.8, -np.inf], dtype=np.float64)
    input_dict = {"name": "isinf_vector_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[0.0, np.inf, -np.inf],
                  [np.nan, 65504.0, 1e5]], dtype=np.float16)
    input_dict = {"name": "isinf_matrix_f16", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([
        [[1.0, -2.0, np.inf], [np.finfo(np.float32).max, -np.inf, 0.0]],
        [[np.nan, 3.14, -1e40], [1e20, 4e38, 5e38]]
    ], dtype=np.float32)
    input_dict = {"name": "isinf_3d_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([[[[np.inf, -np.inf, 1e308, 1e309]],
                   [[-1e309, 0.0, 42.0, -3.14]]]], dtype=np.float64)
    input_dict = {"name": "isinf_4d_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([], dtype=np.float32)
    input_dict = {"name": "isinf_empty_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([[np.nan, -np.nan, np.inf],
                  [np.nan, 0.0, -np.inf]], dtype=np.float64)
    input_dict = {"name": "isinf_mixed_nan_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([1e-45, 1e-38, 1e38, 3.5e38, -3.6e38], dtype=np.float32)
    input_dict = {"name": "isinf_large_range_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array(-np.inf, dtype=np.float64)
    input_dict = {"name": "isinf_scalar_neg_inf_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    data = np.array([0.0, 1e5, -1e5, 65504.0, -65504.0, np.inf,
                     -np.inf, np.nan, 1.0, -2.0, 3.0, -4.0], dtype=np.float16)
    x = data.reshape(2, 3, 2, 1)
    input_dict = {"name": "isinf_high_dim_f16", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    x = np.array([0.0, -0.0, np.finfo(np.float64).max,
                  -np.finfo(np.float64).max, np.inf], dtype=np.float64)
    input_dict = {"name": "isinf_row_vector_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    x = np.arange(25, dtype=np.float32).reshape(5, 5)
    x[0, 0] = np.inf
    x[4, 4] = -np.inf
    input_dict = {"name": "isinf_big2d_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.IsInf"] = tf_raw_ops_isinf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.IsInf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.IsInf'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.IsInf', generated_inputs['tf.raw_ops.IsInf'], lib="tf", suffix=0)
