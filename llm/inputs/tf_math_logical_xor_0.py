
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_logical_xor_inputs():
    list_of_inputs = []

    # Input 1: Two single boolean values
    x = np.array(True, dtype=np.bool_)
    y = np.array(False, dtype=np.bool_)
    name = "xor_single"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tensor and a single boolean
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array(True, dtype=np.bool_)
    name = "xor_tensor_single"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Two tensors of the same shape
    x = np.array([True, True, False, False], dtype=np.bool_)
    y = np.array([True, False, True, False], dtype=np.bool_)
    name = "xor_tensor_tensor"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Two tensors with broadcast-compatible shapes
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    y = np.array([True, False], dtype=np.bool_)
    name = "xor_broadcast"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Two tensors with broadcast-compatible shapes (different example)
    x = np.array([True, False], dtype=np.bool_)
    y = np.array([[True], [False]], dtype=np.bool_)
    name = "xor_broadcast2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensors
    x = np.array([[[True, False], [False, True]], [[False, True], [True, False]]], dtype=np.bool_)
    y = np.array([[[False, True], [True, False]], [[True, False], [False, True]]], dtype=np.bool_)
    name = "xor_3d"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different values
    x = np.array([False, False, False], dtype=np.bool_)
    y = np.array([True, True, True], dtype=np.bool_)
    name = "xor_all_diff"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Same values
    x = np.array([True, True, True], dtype=np.bool_)
    y = np.array([True, True, True], dtype=np.bool_)
    name = "xor_all_same"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty array
    x = np.array([], dtype=np.bool_)
    y = np.array([], dtype=np.bool_)
    name = "xor_empty"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: More complex broadcast
    x = np.array([[[True, False],[True,False]],[[False, True],[False, True]]], dtype=np.bool_)
    y = np.array([True, False], dtype=np.bool_)
    name = "xor_broadcast3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.logical_xor"] = tf_math_logical_xor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.logical_xor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.logical_xor'.")

check_valid('tf.math.logical_xor', generated_inputs['tf.math.logical_xor'], lib="tf", suffix=0)
