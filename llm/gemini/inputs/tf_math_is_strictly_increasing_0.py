
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_is_strictly_increasing_inputs():
    list_of_inputs = []

    # Input 1: Basic increasing sequence
    x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    name = "increasing_sequence"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Decreasing sequence
    x = np.array([5, 4, 3, 2, 1], dtype=np.float32)
    name = "decreasing_sequence"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Sequence with equal elements
    x = np.array([1, 2, 2, 3, 4], dtype=np.float32)
    name = "equal_elements"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty sequence
    x = np.array([], dtype=np.float32)
    name = "empty_sequence"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single element sequence
    x = np.array([5], dtype=np.float32)
    name = "single_element"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Sequence with negative values
    x = np.array([-5, -4, -3, -2, -1], dtype=np.float32)
    name = "negative_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Sequence with mixed positive and negative values
    x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    name = "mixed_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Non-strictly increasing sequence (allows equality)
    x = np.array([1, 2, 3, 3, 4], dtype=np.float32)
    name = "non_strictly_increasing"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Sequence with integers
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    name = "integer_sequence"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Sequence with some duplicates and not increasing
    x = np.array([1, 2, 1, 4, 5], dtype=np.float64)
    name = "not_strictly_increasing2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.is_strictly_increasing"] = tf_math_is_strictly_increasing_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.is_strictly_increasing' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.is_strictly_increasing'.")

check_valid('tf.math.is_strictly_increasing', generated_inputs['tf.math.is_strictly_increasing'], lib="tf", suffix=0)
