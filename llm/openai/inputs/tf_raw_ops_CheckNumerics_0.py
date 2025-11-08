
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_checknumerics_inputs():
    list_of_inputs = []

    tensor = np.array(1.0, dtype=np.float32)
    input_dict = {
        "name": "check_scalar_f32",
        "tensor": tensor,
        "message": "Scalar float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([-3.5, 0.0, 2.2], dtype=np.float64)
    input_dict = {
        "name": "check_vector_f64",
        "tensor": tensor,
        "message": "Vector float64 with negatives"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[1e-4, -1e-3], [2.0, -2.5]], dtype=np.float16)
    input_dict = {
        "name": "check_matrix_f16",
        "tensor": tensor,
        "message": "Matrix float16 small values"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    input_dict = {
        "name": "check_3d_no_nan",
        "tensor": tensor,
        "message": "3D tensor finite values"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[[[1.0, -1.0], [0.0, 123.456]], [[-789.0, 1e-10], [1e20, -1e20]]]], dtype=np.float32)
    input_dict = {
        "name": "check_4d_finite_extremes",
        "tensor": tensor,
        "message": "4D tensor finite extremes"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([], dtype=np.float32)
    input_dict = {
        "name": "check_empty_f32",
        "tensor": tensor,
        "message": "Empty tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[1.0, 2.0, -3.0], [4.5, -5.5, 6.25]], dtype=np.float64)
    input_dict = {
        "name": "check_matrix_f64",
        "tensor": tensor,
        "message": "Matrix float64 finite"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[1e-5, -1e-5], [6e-8, -6e-8]], dtype=np.float16)
    input_dict = {
        "name": "check_subnormal_f16",
        "tensor": tensor,
        "message": "Float16 with small magnitudes"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([1e10, -1e10, 3.1415927, -2.7182818], dtype=np.float32)
    input_dict = {
        "name": "check_vector_f32_large",
        "tensor": tensor,
        "message": "Vector float32 large finite"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[[0.1, 0.2, 0.3], [-0.4, -0.5, 0.6]], [[7.7, -8.8, 9.9], [10.01, -11.11, 12.12]]], dtype=np.float64)
    input_dict = {
        "name": "check_3d_f64",
        "tensor": tensor,
        "message": "3D float64 finite"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([42.0], dtype=np.float32)
    input_dict = {
        "name": "check_single_element",
        "tensor": tensor,
        "message": "Single finite element"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array(-123.456, dtype=np.float64)
    input_dict = {
        "name": "check_scalar_neg_f64",
        "tensor": tensor,
        "message": "Scalar negative float64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.CheckNumerics"] = tf_raw_ops_checknumerics_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.CheckNumerics' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.CheckNumerics'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.CheckNumerics', generated_inputs['tf.raw_ops.CheckNumerics'], lib="tf", suffix=0)
