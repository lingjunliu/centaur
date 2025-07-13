
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_any_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([True, False, True]).astype(np.bool_)
    axis_tensor = np.array(0).astype(np.int32)
    keep_dims_bool = False
    name_str = "any_op_1"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[True, False], [False, True]]).astype(np.bool_)
    axis_tensor = np.array(0).astype(np.int64)
    keep_dims_bool = True
    name_str = "any_op_2"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[True, False], [False, True]]).astype(np.bool_)
    axis_tensor = np.array(1).astype(np.int32)
    keep_dims_bool = False
    name_str = "any_op_3"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[True, False], [False, True]], [[False, True], [True, False]]]).astype(np.bool_)
    axis_tensor = np.array(0).astype(np.int64)
    keep_dims_bool = True
    name_str = "any_op_4"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[True, False], [False, True]], [[False, True], [True, False]]]).astype(np.bool_)
    axis_tensor = np.array(1).astype(np.int32)
    keep_dims_bool = False
    name_str = "any_op_5"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[True, False], [False, True]], [[False, True], [True, False]]]).astype(np.bool_)
    axis_tensor = np.array(2).astype(np.int64)
    keep_dims_bool = True
    name_str = "any_op_6"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[True, False], [False, True]], [[False, True], [True, False]]]).astype(np.bool_)
    axis_tensor = np.array([0, 1]).astype(np.int32)
    keep_dims_bool = False
    name_str = "any_op_7"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[True, False], [False, True]], [[False, True], [True, False]]]).astype(np.bool_)
    axis_tensor = np.array([1, 2]).astype(np.int64)
    keep_dims_bool = True
    name_str = "any_op_8"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array(False).astype(np.bool_)
    axis_tensor = np.array(0).astype(np.int32)
    keep_dims_bool = False
    name_str = "any_op_9"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([True]).astype(np.bool_)
    axis_tensor = np.array(0).astype(np.int64)
    keep_dims_bool = True
    name_str = "any_op_10"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Any' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Any'.")

check_valid('tf.raw_ops.Any', generated_inputs['tf.raw_ops.Any'], lib="tf", suffix=0)
