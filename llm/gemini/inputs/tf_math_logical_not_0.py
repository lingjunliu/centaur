
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_logical_not_inputs():
    list_of_inputs = []

    # Input 1: Scalar boolean
    x = np.array(True, dtype=np.bool_)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D boolean array
    x = np.array([True, False, True, False], dtype=np.bool_)
    name = "bool_array_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D boolean array
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    name = "bool_array_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D boolean array
    x = np.array([[[True, False], [False, True]], [[False, True], [True, False]]], dtype=np.bool_)
    name = "bool_array_3d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty boolean array
    x = np.array([], dtype=np.bool_)
    name = "empty_array"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Boolean array with only True values
    x = np.array([True, True, True], dtype=np.bool_)
    name = "all_true"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Boolean array with only False values
    x = np.array([False, False, False], dtype=np.bool_)
    name = "all_false"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different name
    x = np.array([True, False], dtype=np.bool_)
    name = "different_name"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger boolean array
    x = np.random.choice([True, False], size=(10, 10), replace=True).astype(np.bool_)
    name = "large_bool_array"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: name as an empty string
    x = np.array(True, dtype=np.bool_)
    name = ""
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.logical_not"] = tf_math_logical_not_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.logical_not' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.logical_not'.")

check_valid('tf.math.logical_not', generated_inputs['tf.math.logical_not'], lib="tf", suffix=0)
