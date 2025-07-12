
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_is_non_decreasing_inputs():
    list_of_inputs = []

    # Input 1: Non-decreasing 1D array
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    name = "non_decreasing_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Decreasing 1D array
    x = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    name = "decreasing_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Non-decreasing with duplicates
    x = np.array([1.0, 1.0, 2.0, 2.0, 3.0], dtype=np.float32)
    name = "non_decreasing_duplicates"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mixed increasing and decreasing
    x = np.array([1.0, 2.0, 1.0, 2.0, 3.0], dtype=np.float32)
    name = "mixed_increasing_decreasing"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single element array
    x = np.array([5.0], dtype=np.float32)
    name = "single_element"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty array
    x = np.array([], dtype=np.float32)
    name = "empty_array"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Non-decreasing 2D array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    name = "non_decreasing_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Decreasing 2D array
    x = np.array([[5.0, 4.0], [3.0, 2.0]], dtype=np.float32)
    name = "decreasing_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Mixed 2D array
    x = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    name = "mixed_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Non-decreasing int array
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    name = "non_decreasing_int"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Negative values
    x = np.array([-5, -4, -3, -2, -1], dtype=np.int32)
    name = "negative_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Mixed positive and negative
    x = np.array([-1, 0, 1, 2, 3], dtype=np.int32)
    name = "mixed_pos_neg"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.is_non_decreasing"] = tf_math_is_non_decreasing_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.is_non_decreasing' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.is_non_decreasing'.")

check_valid('tf.math.is_non_decreasing', generated_inputs['tf.math.is_non_decreasing'], lib="tf", suffix=0)
