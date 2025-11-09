
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_erf_inputs():
    list_of_inputs = []

    x = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, -2.0]], dtype=np.float32)
    input_dict = {"name": "erf_matrix_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-5.0, -2.5, 0.0], [1.25, 3.75, 10.0]], dtype=np.float64)
    input_dict = {"name": "erf_matrix_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(0.5, dtype=np.float16)
    input_dict = {"name": "erf_scalar_f16", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    input_dict = {"name": "erf_vector_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    rs = np.random.RandomState(0)
    x = rs.normal(size=(2, 2, 3)).astype(np.float64)
    input_dict = {"name": "erf_3d_random_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[[-10.0, 0.0, 10.0]]], [[[20.0, -20.0, 0.0]]]], dtype=np.float32)
    input_dict = {"name": "erf_4d_large_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    input_dict = {"name": "erf_empty_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.nan, np.inf, -np.inf, -0.0, 0.0], dtype=np.float64)
    input_dict = {"name": "erf_specials_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.arange(0, 8, dtype=np.float16).reshape(1, 2, 2, 2, 1)
    input_dict = {"name": "erf_5d_f16", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.linspace(-2.0, 2.0, num=24, dtype=np.float32).reshape(2, 3, 4)
    input_dict = {"name": "erf_3d_linspace_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-7.5, -0.0, 0.0, 7.5], dtype=np.float64)
    input_dict = {"name": "erf_vector_precise_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-8.0, -4.0, -2.0], [2.0, 4.0, 8.0]], dtype=np.float16)
    input_dict = {"name": "erf_matrix_f16", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Erf"] = tf_raw_ops_erf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Erf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Erf'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Erf', generated_inputs['tf.raw_ops.Erf'], lib="tf", suffix=0)
