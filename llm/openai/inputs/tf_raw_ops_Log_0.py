
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_log_inputs():
    list_of_inputs = []

    x = np.array([0.0, 0.5, 1.0, 5.0], dtype=np.float32)
    input_dict = {"name": "log_case_1", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1e-3, 2.5], [10.0, 100.0]], dtype=np.float64)
    input_dict = {"name": "log_case_2", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[-1.0, 0.0], [2.0, 3.5]], [[-5.0, 7.0], [0.1, -0.2]]], dtype=np.float16)
    input_dict = {"name": "log_case_3", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1+1j, -1+0j, 0+2j, -3-4j], dtype=np.complex64)
    input_dict = {"name": "log_case_4", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1-1j, 2+0.5j], [0.001+3j, -2-2j]], dtype=np.complex128)
    input_dict = {"name": "log_case_5", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e10, 1e20, 1e-10, 3.14159265], dtype=np.float32)
    input_dict = {"name": "log_case_6", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-300, 1e-200, 1e-100, 1e-50], dtype=np.float64)
    input_dict = {"name": "log_case_7", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -0.5, -10.0, -1e-6], dtype=np.float32)
    input_dict = {"name": "log_case_8", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(2.718281828, dtype=np.float64)
    input_dict = {"name": "log_case_9_scalar_float64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.ones((2, 3, 4, 5), dtype=np.float32) * 2.0
    input_dict = {"name": "log_case_10_4d_float32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(1+0j, dtype=np.complex64)
    input_dict = {"name": "log_case_11_scalar_complex64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    input_dict = {"name": "log_case_12_empty_float32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0.0, -0.0], [-np.finfo(np.float32).tiny, np.finfo(np.float32).tiny]], dtype=np.float32)
    input_dict = {"name": "log_case_13_edge_float32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.linspace(0.1, 10.0, 15, dtype=np.float64).reshape(3, 5)
    input_dict = {"name": "log_case_14_linspace_float64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Log"] = tf_raw_ops_log_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Log' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Log'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Log', generated_inputs['tf.raw_ops.Log'], lib="tf", suffix=0)
