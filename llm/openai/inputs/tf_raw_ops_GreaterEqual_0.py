
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import torch
import copy

def tf_raw_ops_greater_equal_inputs():
    list_of_inputs = []

    x = np.array([5, 4, 6, 7], dtype=np.int32)
    y = np.array([5, 2, 5, 10], dtype=np.int32)
    input_dict = {"name": "ge_case_1", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([5, 4, 6, 7], dtype=np.int32)
    y = np.array([5], dtype=np.int32)
    input_dict = {"name": "ge_case_2", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]], dtype=np.float32)
    y = np.array([[2.0, 2.0, 2.0],
                  [3.0, 6.0, 6.0]], dtype=np.float32)
    input_dict = {"name": "ge_case_3", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1, 0, 2],
                  [5, -5, 10]], dtype=np.int64)
    y = np.array([0, 0, 1], dtype=np.int64)
    input_dict = {"name": "ge_case_4", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[ -1.5,  0.0,  2.5, -3.2]],
                  [[ 10.1, -5.0,  0.0,  7.7]]], dtype=np.float64)
    y = np.array([[[ -2.0],
                   [  0.0],
                   [  5.0]]], dtype=np.float64)
    input_dict = {"name": "ge_case_5", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-128, -1, 127], dtype=np.int8)
    y = np.array(0, dtype=np.int8)
    input_dict = {"name": "ge_case_6", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0, 255],
                  [128, 64]], dtype=np.uint8)
    y = np.array([100], dtype=np.uint8)
    input_dict = {"name": "ge_case_7", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)
    y = np.array([0.0, np.inf, -1.0, np.nan], dtype=np.float32)
    input_dict = {"name": "ge_case_8", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.5, -2.0, 0.0, 65504.0]], dtype=np.float16)
    y = np.array(1.0, dtype=np.float16)
    input_dict = {"name": "ge_case_9", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1],
                  [2],
                  [3]], dtype=np.int32)
    y = np.array([[0, 1, 2, 3]], dtype=np.int32)
    input_dict = {"name": "ge_case_10", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, -2],
                   [3, -4]],
                  [[5, -6],
                   [7, -8]]], dtype=np.int16)
    y = np.array([[[0],
                   [2]]], dtype=np.int16)
    input_dict = {"name": "ge_case_11", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-3, dtype=np.int64)
    y = np.array(-3, dtype=np.int64)
    input_dict = {"name": "ge_case_12", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-5, dtype=np.int32)
    y = np.array([[-10, -5, 0, 5]], dtype=np.int32)
    input_dict = {"name": "ge_case_13", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.GreaterEqual"] = tf_raw_ops_greater_equal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.GreaterEqual' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GreaterEqual'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.GreaterEqual', generated_inputs['tf.raw_ops.GreaterEqual'], lib="tf", suffix=0)
