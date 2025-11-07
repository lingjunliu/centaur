
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_Where_inputs():
    list_of_inputs = []

    condition = np.array([[True, False], [True, False]], dtype=np.bool_)
    input_dict = {'name': 'where_test_1', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = np.array([[[True, False], [True, False]], [[False, True], [False, True]]], dtype=np.bool_)
    input_dict = {'name': 'where_test_2', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = np.array([[[1.5, 0.0], [-0.5, 0.0]], [[0.0, 0.25], [0.0, 0.75]]], dtype=np.float32)
    input_dict = {'name': 'where_test_3', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = np.array([[[1.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.5j, 0.0 + 0.0j]], [[0.0 + 0.0j, 0.25 + 1.5j], [0.0 + 0.0j, 0.75 + 0.0j]]], dtype=np.complex64)
    input_dict = {'name': 'where_test_4', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = np.array([[1, 2, 3], [4, 0, 6]], dtype=np.int32)
    input_dict = {'name': 'where_test_5', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = np.array([[-1, -2, -3], [-4, 0, -6]], dtype=np.int32)
    input_dict = {'name': 'where_test_6', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float64)
    input_dict = {'name': 'where_test_7', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = np.array([[[True, True], [False, False]], [[True, False], [False, True]]], dtype=np.bool_)
    input_dict = {'name': 'where_test_8', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = np.array([1, 0, 1, 0, 1], dtype=np.int64)
    input_dict = {'name': 'where_test_9', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = np.array([[[True]], [[False]]], dtype=np.bool_)
    input_dict = {'name': 'where_test_10', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Where"] = tf_raw_ops_Where_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Where' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Where'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Where', generated_inputs['tf.raw_ops.Where'], lib="tf", suffix=0)
