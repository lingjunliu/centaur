
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

np.random.seed(42)
tf.random.set_seed(42)

def tf_raw_ops_any_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = np.array([True, False, False, True], dtype=np.bool_)
    axis = np.array(0, dtype=np.int32)
    keep_dims = False
    name = "any_case_1"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 2
    input_arr = np.array([[True, False, True], [False, False, True]], dtype=np.bool_)
    axis = np.array([0], dtype=np.int64)
    keep_dims = False
    name = "any_case_2"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 3
    input_arr = np.array([[False, False, True], [True, False, False]], dtype=np.bool_)
    axis = np.array([1], dtype=np.int32)
    keep_dims = True
    name = "any_case_3"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 4
    input_arr = np.array([[[True, False], [False, False]],
                          [[True, True], [False, True]]], dtype=np.bool_)
    axis = np.array([0, 2], dtype=np.int64)
    keep_dims = False
    name = "any_case_4"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 5
    input_arr = (np.random.rand(2, 3, 4) > 0.7).astype(np.bool_)
    axis = np.array(-1, dtype=np.int32)
    keep_dims = True
    name = "any_case_5"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 6
    input_arr = np.array(True, dtype=np.bool_)
    axis = np.array([], dtype=np.int32)
    keep_dims = False
    name = "any_case_6_scalar_empty_axis"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 7
    input_arr = np.zeros((2, 0, 3, 1), dtype=np.bool_)
    axis = np.array([1], dtype=np.int64)
    keep_dims = False
    name = "any_case_7_zero_size_dim"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 8
    input_arr = np.array([], dtype=np.bool_)
    axis = np.array(0, dtype=np.int32)
    keep_dims = True
    name = "any_case_8_empty_vector"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 9
    input_arr = (np.random.rand(3, 4, 5) > 0.8).astype(np.bool_)
    axis = np.array([0, 1, 2], dtype=np.int64)
    keep_dims = False
    name = "any_case_9_reduce_all_axes"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 10
    input_arr = (np.random.rand(3, 4, 5) > 0.3).astype(np.bool_)
    axis = np.array([-2], dtype=np.int32)
    keep_dims = True
    name = "any_case_10_negative_axis"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 11
    input_arr = np.array([[False]], dtype=np.bool_)
    axis = np.array([0, 1], dtype=np.int32)
    keep_dims = False
    name = "any_case_11_single_element_all_axes"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 12
    input_arr = np.array([[True, False, True, False, False, True],
                          [False, False, False, False, True, False],
                          [True, True, False, False, False, False],
                          [False, True, True, False, False, False],
                          [False, False, True, True, False, True]], dtype=np.bool_)
    axis = np.array(-1, dtype=np.int64)
    keep_dims = False
    name = "any_case_12_last_axis"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    return list_of_inputs

generated_inputs["tf.raw_ops.Any"] = tf_raw_ops_any_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Any' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Any'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Any', generated_inputs['tf.raw_ops.Any'], lib="tf", suffix=0)
