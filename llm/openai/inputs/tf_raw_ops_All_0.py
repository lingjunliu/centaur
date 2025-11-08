
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_all_inputs():
    list_of_inputs = []

    input_arr = np.array([True, True, False, True, True], dtype=bool)
    axis = np.int32(0)
    keep_dims = False
    name = "all_case_1"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[True, False, True, True],
                          [True, True, True, False],
                          [False, True, True, True]], dtype=bool)
    axis = np.array([0], dtype=np.int64)
    keep_dims = True
    name = "all_case_2"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[True, False, True, True],
                           [True, True, True, True],
                           [True, True, False, True]],
                          [[True, True, True, True],
                           [True, False, True, True],
                           [True, True, True, True]]], dtype=bool)
    axis = np.array([-1], dtype=np.int32)
    keep_dims = False
    name = "all_case_3"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[True, True, False],
                           [True, False, True],
                           [True, True, True],
                           [False, True, True]],
                          [[True, True, True],
                           [True, True, False],
                           [True, False, True],
                           [True, True, True]]], dtype=bool)
    axis = np.array([0, 2], dtype=np.int64)
    keep_dims = True
    name = "all_case_4"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array(False, dtype=bool)
    axis = np.array([], dtype=np.int32)
    keep_dims = False
    name = "all_case_5_scalar_empty_axis"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[[True], [True], [False]]],
                          [[[True], [True], [True]]]], dtype=bool)
    axis = np.int64(2)
    keep_dims = True
    name = "all_case_6"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[[True, True],
                            [True, True]],
                           [[True, True],
                            [True, False]]],
                          [[[True, True],
                            [True, True]],
                           [[True, True],
                            [True, True]]]], dtype=bool)
    axis = np.array([0, 1, 2, 3], dtype=np.int32)
    keep_dims = False
    name = "all_case_7_reduce_all_axes"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[True, False, True, True, True]], dtype=bool)
    axis = np.array([-2], dtype=np.int64)
    keep_dims = True
    name = "all_case_8_negative_axis_scalar"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[True, True, True],
                           [True, True, True],
                           [True, True, True]],
                          [[True, True, True],
                           [True, True, True],
                           [True, True, True]],
                          [[True, True, True],
                           [True, True, True],
                           [False, True, True]]], dtype=bool)
    axis = np.int32(-3)
    keep_dims = False
    name = "all_case_9_negative_scalar_axis"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[[True], [True], [False]],
                           [[True], [True], [True]]],
                          [[[True], [False], [True]],
                           [[True], [True], [True]]]], dtype=bool)
    axis = np.array([1, 3], dtype=np.int64)
    keep_dims = True
    name = "all_case_10_multi_axes"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[True, True],
                          [True, True]], dtype=bool)
    axis = np.array([0, 1], dtype=np.int32)
    keep_dims = True
    name = "all_case_11_reduce_all_keepdims"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[True, False],
                           [True, True],
                           [True, True],
                           [True, True]],
                          [[True, True],
                           [False, True],
                           [True, True],
                           [True, True]],
                          [[True, True],
                           [True, True],
                           [True, False],
                           [True, True]],
                          [[True, True],
                           [True, True],
                           [True, True],
                           [True, True]]], dtype=bool)
    axis = np.array([1], dtype=np.int64)
    keep_dims = False
    name = "all_case_12_axis1"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.All"] = tf_raw_ops_all_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.All' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.All'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.All', generated_inputs['tf.raw_ops.All'], lib="tf", suffix=0)
