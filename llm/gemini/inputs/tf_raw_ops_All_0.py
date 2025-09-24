
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_all_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    input_tensor = np.array([[True, True], [False, True]], dtype=np.bool_)
    axis_tensor = np.array([0])
    keep_dims_val = False
    name_val = "all_op_1"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Keep dimensions
    input_tensor = np.array([[True, True], [False, True]], dtype=np.bool_)
    axis_tensor = np.array([0])
    keep_dims_val = True
    name_val = "all_op_2"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple axis
    input_tensor = np.array([[[True, True], [False, True]], [[True, False], [True, True]]], dtype=np.bool_)
    axis_tensor = np.array([0, 1])
    keep_dims_val = False
    name_val = "all_op_3"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative axis
    input_tensor = np.array([[True, True], [False, True]], dtype=np.bool_)
    axis_tensor = np.array([-1])
    keep_dims_val = False
    name_val = "all_op_4"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: All False
    input_tensor = np.array([[False, False], [False, False]], dtype=np.bool_)
    axis_tensor = np.array([0])
    keep_dims_val = False
    name_val = "all_op_6"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All True
    input_tensor = np.array([[True, True], [True, True]], dtype=np.bool_)
    axis_tensor = np.array([0])
    keep_dims_val = False
    name_val = "all_op_7"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor, keep_dims=True, multiple axes
    input_tensor = np.array([[[True, True], [False, True]], [[True, False], [True, True]]], dtype=np.bool_)
    axis_tensor = np.array([0, 2])
    keep_dims_val = True
    name_val = "all_op_8"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single element tensor
    input_tensor = np.array([[True]], dtype=np.bool_)
    axis_tensor = np.array([0])
    keep_dims_val = False
    name_val = "all_op_9"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int64 axis
    input_tensor = np.array([[True, True], [False, True]], dtype=np.bool_)
    axis_tensor = np.array([0], dtype=np.int64)
    keep_dims_val = False
    name_val = "all_op_10"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int32 axis
    input_tensor = np.array([[True, True], [False, True]], dtype=np.bool_)
    axis_tensor = np.array([0], dtype=np.int32)
    keep_dims_val = False
    name_val = "all_op_11"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D input tensor
    input_tensor = np.array([True, False, True], dtype=np.bool_)
    axis_tensor = np.array([0])
    keep_dims_val = False
    name_val = "all_op_12"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Empty tensor, axis = None
    input_tensor = np.array([], dtype=np.bool_)
    axis_tensor = np.array([0])
    keep_dims_val = False
    name_val = "all_op_13"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.All"] = tf_raw_ops_all_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.All' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.All'.")

check_valid('tf.raw_ops.All', generated_inputs['tf.raw_ops.All'], lib="tf", suffix=0)
