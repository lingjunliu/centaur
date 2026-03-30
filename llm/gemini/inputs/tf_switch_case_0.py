
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_switch_case_inputs():
    list_of_inputs = []

    # Input 1
    branch_index = tf.constant(0, dtype=tf.int32)
    branch_fns = [lambda: tf.constant(10), lambda: tf.constant(20)]
    default = [tf.constant(0)]
    name = "switch_case_1"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": [tf.constant(0)], "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    branch_index = tf.constant(1, dtype=tf.int32)
    branch_fns = [lambda: tf.constant(10), lambda: tf.constant(20)]
    default = [tf.constant(0)]
    name = "switch_case_2"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": [tf.constant(0)], "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    branch_index = tf.constant(2, dtype=tf.int32)
    branch_fns = [lambda: tf.constant(10), lambda: tf.constant(20)]
    default = [tf.constant(30)]
    name = "switch_case_3"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": [tf.constant(30)], "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    branch_index = tf.constant(0, dtype=tf.int32)
    branch_fns = [lambda: tf.constant([1, 2]), lambda: tf.constant([3, 4])]
    default = [tf.constant([0, 0])]
    name = "switch_case_4"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": [tf.constant([0, 0])], "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    branch_index = tf.constant(1, dtype=tf.int32)
    branch_fns = [lambda: tf.constant([1, 2]), lambda: tf.constant([3, 4])]
    default = [tf.constant([0, 0])]
    name = "switch_case_5"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": [tf.constant([0, 0])], "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    branch_index = tf.constant(0, dtype=tf.int32)
    branch_fns = [lambda: tf.constant([[1, 2], [3, 4]]), lambda: tf.constant([[5, 6], [7, 8]])]
    default = [tf.constant([[0, 0], [0, 0]])]
    name = "switch_case_6"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": [tf.constant([[0, 0], [0, 0]])], "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    branch_index = tf.constant(1, dtype=tf.int32)
    branch_fns = [lambda: tf.constant([[1, 2], [3, 4]]), lambda: tf.constant([[5, 6], [7, 8]])]
    default = [tf.constant([[0, 0], [0, 0]])]
    name = "switch_case_7"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": [tf.constant([[0, 0], [0, 0]])], "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    branch_index = tf.constant(2, dtype=tf.int32)
    branch_fns = [lambda: tf.constant(10), lambda: tf.constant(20), lambda: tf.constant(30)]
    default = [tf.constant(0)]
    name = "switch_case_8"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": [tf.constant(0)], "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    branch_index = tf.constant(0, dtype=tf.int32)
    branch_fns = [lambda: (tf.constant(1), tf.constant(2)), lambda: (tf.constant(3), tf.constant(4))]
    default = [tf.constant(0), tf.constant(0)]
    name = "switch_case_9"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": [tf.constant(0), tf.constant(0)], "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    branch_index = tf.constant(1, dtype=tf.int32)
    branch_fns = [lambda: (tf.constant(1), tf.constant(2)), lambda: (tf.constant(3), tf.constant(4))]
    default = [tf.constant(0), tf.constant(0)]
    name = "switch_case_10"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": [tf.constant(0), tf.constant(0)], "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    branch_index = tf.constant(0, dtype=tf.int32)
    branch_fns = [lambda: tf.constant(10)]
    default = [tf.constant(0)]
    name = "switch_case_11"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": [tf.constant(0)], "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    branch_index = tf.constant(0, dtype=tf.int32)
    branch_fns = [lambda: tf.constant(10), lambda: tf.constant(20), lambda: tf.constant(30)]
    default = [tf.constant(0)]
    name = "switch_case_12"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": [tf.constant(0)], "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13
    branch_index = tf.constant(0, dtype=tf.int32)
    branch_fns = [lambda: tf.constant(10), lambda: tf.constant(20)]
    default = [tf.constant([0,0])]
    name = "switch_case_13"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": [tf.constant([0,0])], "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14
    branch_index = tf.constant(0, dtype=tf.int32)
    branch_fns = [lambda: tf.constant(10), lambda: tf.constant(20)]
    default = [tf.constant([[0,0],[0,0]])]
    name = "switch_case_14"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": [tf.constant([[0,0],[0,0]])], "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.switch_case"] = tf_switch_case_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.switch_case' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.switch_case'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.switch_case', generated_inputs['tf.switch_case'], lib="tf", suffix=0)
