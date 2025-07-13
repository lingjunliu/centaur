
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_switch_case_inputs():
    list_of_inputs = []

    # Input 1: Basic case with default
    branch_index = tf.constant(0, dtype=tf.int32)
    branch_fns = [lambda: tf.constant(10), lambda: tf.constant(20)]
    default = [lambda: tf.constant(30)]
    name = "switch_case_1"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": default, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: No default, index out of range
    branch_index = tf.constant(2, dtype=tf.int32)
    branch_fns = [lambda: tf.constant(10), lambda: tf.constant(20)]
    default = []
    name = "switch_case_2"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": default, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Nested structure, with default
    branch_index = tf.constant(1, dtype=tf.int32)
    branch_fns = [lambda: (tf.constant(1), tf.constant(2)), lambda: (tf.constant(3), tf.constant(4))]
    default = [lambda: (tf.constant(5), tf.constant(6))]
    name = "switch_case_3"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": default, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List of callables, no default
    branch_index = tf.constant(0, dtype=tf.int32)
    branch_fns = [lambda: tf.constant([1, 2]), lambda: tf.constant([3, 4])]
    default = []
    name = "switch_case_4"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": default, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty branch_fns, with default
    branch_index = tf.constant(0, dtype=tf.int32)
    branch_fns = []
    default = [lambda: tf.constant(100)]
    name = "switch_case_5"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": default, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Branch index is a scalar tensor
    branch_index = tf.constant(1, dtype=tf.int32)
    branch_fns = [lambda: tf.constant(10), lambda: tf.constant(20)]
    default = [lambda: tf.constant(30)]
    name = "switch_case_6"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": default, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Default is None
    branch_index = tf.constant(0, dtype=tf.int32)
    branch_fns = [lambda: tf.constant(1), lambda: tf.constant(2)]
    default = []
    name = "switch_case_7"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": default, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Same return type for all branches
    branch_index = tf.constant(1, dtype=tf.int32)
    branch_fns = [lambda: tf.constant([1, 2]), lambda: tf.constant([3, 4])]
    default = [lambda: tf.constant([5, 6])]
    name = "switch_case_8"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": default, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Default is a numpy array converted to tensor
    branch_index = tf.constant(0, dtype=tf.int32)
    branch_fns = [lambda: tf.constant(1), lambda: tf.constant(2)]
    default = [lambda: tf.convert_to_tensor(np.array([3, 4]), dtype=tf.int32)]
    name = "switch_case_9"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": default, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Name with special characters
    branch_index = tf.constant(0, dtype=tf.int32)
    branch_fns = [lambda: tf.constant(1), lambda: tf.constant(2)]
    default = [lambda: tf.convert_to_tensor(np.array([3, 4]), dtype=tf.int32)]
    name = "switch_case-10_test"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": default, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.switch_case"] = tf_switch_case_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.switch_case' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.switch_case'.")

check_valid('tf.switch_case', generated_inputs['tf.switch_case'], lib="tf", suffix=0)
