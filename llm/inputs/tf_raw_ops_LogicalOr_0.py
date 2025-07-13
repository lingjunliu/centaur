
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_logical_or_inputs():
    list_of_inputs = []

    # Input 1: Basic case, two bool tensors
    x = np.array([True, False, True]).astype(np.bool_)
    y = np.array([False, True, True]).astype(np.bool_)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Broadcasting with a scalar
    x = np.array([True, False, True]).astype(np.bool_)
    y = np.array(True).astype(np.bool_)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Two dimensional arrays
    x = np.array([[True, False], [False, True]]).astype(np.bool_)
    y = np.array([[False, True], [True, False]]).astype(np.bool_)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting with different shapes
    x = np.array([[True, False]]).astype(np.bool_)
    y = np.array([[True], [False]]).astype(np.bool_)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Three dimensional array
    x = np.array([[[True, False], [False, True]], [[False, True], [True, False]]]).astype(np.bool_)
    y = np.array([[[False, True], [True, False]], [[True, False], [False, True]]]).astype(np.bool_)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Name provided
    x = np.array([True, False]).astype(np.bool_)
    y = np.array([False, True]).astype(np.bool_)
    input_dict = {"x": x, "y": y, "name": "logical_or_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: All True values
    x = np.array([True, True, True]).astype(np.bool_)
    y = np.array([True, True, True]).astype(np.bool_)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All False values
    x = np.array([False, False, False]).astype(np.bool_)
    y = np.array([False, False, False]).astype(np.bool_)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different sizes, but broadcastable
    x = np.array([True]).astype(np.bool_)
    y = np.array([False, True, False]).astype(np.bool_)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex shape
    x = np.array([[[True, False], [True, True]], [[False, False], [True, False]]]).astype(np.bool_)
    y = np.array(True).astype(np.bool_)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_logical_or_inputs()
generated_inputs["tf.raw_ops.LogicalOr"] = []
for input_dict in inputs:
    x_np = input_dict["x"]
    y_np = input_dict["y"]
    x_tensor = tf.convert_to_tensor(x_np)
    y_tensor = tf.convert_to_tensor(y_np)
    generated_inputs["tf.raw_ops.LogicalOr"].append({
        "x": x_tensor,
        "y": y_tensor,
        "name": input_dict["name"]
    })

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.LogicalOr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LogicalOr'.")

check_valid('tf.raw_ops.LogicalOr', generated_inputs['tf.raw_ops.LogicalOr'], lib="tf", suffix=0)
