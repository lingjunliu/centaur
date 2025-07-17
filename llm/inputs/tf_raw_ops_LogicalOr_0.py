
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_logical_or_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two single boolean values
    x = np.array(True, dtype=np.bool_)
    y = np.array(False, dtype=np.bool_)
    input_dict = {"x": x, "y": y, "name": "basic_case"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Two boolean tensors of the same shape
    x = np.array([True, False, True], dtype=np.bool_)
    y = np.array([False, True, False], dtype=np.bool_)
    input_dict = {"x": x, "y": y, "name": "same_shape"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: One boolean tensor and one single boolean value
    x = np.array([False, True, False, True], dtype=np.bool_)
    y = np.array(True, dtype=np.bool_)
    input_dict = {"x": x, "y": y, "name": "tensor_and_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Two boolean tensors with broadcast-compatible shapes
    x = np.array([[True, False]], dtype=np.bool_)
    y = np.array([[False], [True]], dtype=np.bool_)
    input_dict = {"x": x, "y": y, "name": "broadcasting"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional tensors
    x = np.array([[[True, False], [False, True]], [[False, False], [True, True]]], dtype=np.bool_)
    y = np.array([[[False, True], [True, False]], [[True, True], [False, False]]], dtype=np.bool_)
    input_dict = {"x": x, "y": y, "name": "multi_dimensional"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shapes that are broadcastable
    x = np.array([True, False, True], dtype=np.bool_)
    y = np.array([[False], [True], [False]], dtype=np.bool_)
    input_dict = {"x": x, "y": y, "name": "broadcastable_different_shapes"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: More broadcasting with different dimensions
    x = np.array([[[True], [False]]], dtype=np.bool_)
    y = np.array([[[False, True]]], dtype=np.bool_)
    input_dict = {"x": x, "y": y, "name": "more_broadcasting"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All False values
    x = np.array([False, False, False], dtype=np.bool_)
    y = np.array([False, False, False], dtype=np.bool_)
    input_dict = {"x": x, "y": y, "name": "all_false"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All True values
    x = np.array([True, True, True], dtype=np.bool_)
    y = np.array([True, True, True], dtype=np.bool_)
    input_dict = {"x": x, "y": y, "name": "all_true"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different shapes
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    y = np.array([True, False], dtype=np.bool_)
    input_dict = {"x": x, "y": y, "name": "diff_shape"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.LogicalOr"] = tf_raw_ops_logical_or_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.LogicalOr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LogicalOr'.")

check_valid('tf.raw_ops.LogicalOr', generated_inputs['tf.raw_ops.LogicalOr'], lib="tf", suffix=0)
