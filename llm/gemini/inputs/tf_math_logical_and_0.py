
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_logical_and_inputs():
    list_of_inputs = []

    # Input 1: Two single boolean values
    x = np.array(True, dtype=np.bool_)
    y = np.array(False, dtype=np.bool_)
    name = "logical_and_1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: One boolean tensor and one single boolean value
    x = np.array([False, True, True, False], dtype=np.bool_)
    y = np.array(True, dtype=np.bool_)
    name = "logical_and_2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Two boolean tensors of the same shape
    x = np.array([False, False, True, True], dtype=np.bool_)
    y = np.array([False, True, False, True], dtype=np.bool_)
    name = "logical_and_3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting example
    x = np.array([[True, False]], dtype=np.bool_)
    y = np.array([[True], [False]], dtype=np.bool_)
    name = "logical_and_4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Another example with broadcasting
    x = np.array([True, False, True], dtype=np.bool_)
    y = np.array([[True], [True], [False]], dtype=np.bool_)
    name = "logical_and_5"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D tensors
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    y = np.array([[True, True], [False, False]], dtype=np.bool_)
    name = "logical_and_6"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensors
    x = np.array([[[True, False], [False, True]], [[True, True], [False, False]]], dtype=np.bool_)
    y = np.array([[[False, True], [True, False]], [[False, False], [True, True]]], dtype=np.bool_)
    name = "logical_and_7"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: One tensor, one scalar False
    x = np.array([True, True, False], dtype=np.bool_)
    y = np.array(False, dtype=np.bool_)
    name = "logical_and_8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty tensor
    x = np.array([], dtype=np.bool_)
    y = np.array([], dtype=np.bool_)
    name = "logical_and_9"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: scalar tensors
    x = np.array(np.True_, dtype=np.bool_)
    y = np.array(np.False_, dtype=np.bool_)
    name = "logical_and_10"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.logical_and"] = tf_math_logical_and_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.logical_and' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.logical_and'.")

check_valid('tf.math.logical_and', generated_inputs['tf.math.logical_and'], lib="tf", suffix=0)
