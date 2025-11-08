
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Exp_inputs():
    list_of_inputs = []

    x = np.array(0.0, dtype=np.float32)
    input_dict = {"name": "exp_input_1", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([2.0, 8.0, -1.5, 0.0], dtype=np.float64)
    input_dict = {"name": "exp_input_2", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float16)
    input_dict = {"name": "exp_input_3", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[-3.0, -0.5], [0.5, 1.5]], [[2.5, -2.5], [4.0, -4.0]]], dtype=np.float32)
    input_dict = {"name": "exp_input_4", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[[0.0, 1.0, -1.0]], [[2.0, -2.0, 3.0]]]], dtype=np.float64)
    input_dict = {"name": "exp_input_5", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(1.0 + 1.0j, dtype=np.complex64)
    input_dict = {"name": "exp_input_6", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0 + 2.0j, 0.0 - 0.5j, 3.0 + 0.0j], dtype=np.complex128)
    input_dict = {"name": "exp_input_7", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([15.0, -15.0, 5.5, -7.25], dtype=np.float32)
    input_dict = {"name": "exp_input_8", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10.0, -10.0, 20.0, -20.0], dtype=np.float16)
    input_dict = {"name": "exp_input_9", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.nan, np.inf, -np.inf, 1.0, -1.0], dtype=np.float32)
    input_dict = {"name": "exp_input_10", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.nan + 1j, 1.0 + np.nan * 1j, np.inf + 0j, -np.inf + 2j], dtype=np.complex64)
    input_dict = {"name": "exp_input_11", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    input_dict = {"name": "exp_input_12", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0 + 0.0j, -1.0 + 1.0j], [0.5 - 0.5j, 2.0 + 2.0j]]], dtype=np.complex128)
    input_dict = {"name": "exp_input_13", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Exp"] = tf_raw_ops_Exp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Exp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Exp'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Exp', generated_inputs['tf.raw_ops.Exp'], lib="tf", suffix=0)
