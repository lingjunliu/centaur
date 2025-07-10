
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_is_non_decreasing_inputs():
    list_of_inputs = []

    # Input 1: Non-decreasing sequence
    x = tf.constant(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
    name = "non_decreasing_test"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Decreasing sequence
    x = tf.constant(np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32))
    name = "decreasing_test"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Sequence with equal elements
    x = tf.constant(np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32))
    name = "equal_elements_test"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mixed increasing and equal
    x = tf.constant(np.array([1.0, 2.0, 2.0, 3.0, 4.0], dtype=np.float32))
    name = "mixed_increasing_test"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Mixed decreasing and equal
    x = tf.constant(np.array([4.0, 3.0, 3.0, 2.0, 1.0], dtype=np.float32))
    name = "mixed_decreasing_test"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small sequence
    x = tf.constant(np.array([1.0, 2.0], dtype=np.float32))
    name = "small_sequence_test"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: One element sequence
    x = tf.constant(np.array([1.0], dtype=np.float32))
    name = "one_element_test"
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
