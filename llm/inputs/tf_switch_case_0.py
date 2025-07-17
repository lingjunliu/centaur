
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_switch_case_inputs():
    list_of_inputs = []

    # Input 1
    branch_index = tf.constant(0)
    branch_fns = [lambda: tf.constant(1).numpy(), lambda: tf.constant(2).numpy()]
    default = [lambda: tf.constant(-1).numpy()]
    name = "case1"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": default, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    branch_index = tf.constant(1)
    branch_fns = [lambda: tf.constant([1, 2]).numpy(), lambda: tf.constant([3, 4]).numpy()]
    default = [lambda: tf.constant([-1, -2]).numpy()]
    name = "case2"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": default, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    branch_index = tf.constant(2)
    branch_fns = [lambda: tf.constant(1).numpy(), lambda: tf.constant(2).numpy(), lambda: tf.constant(3).numpy()]
    default = [lambda: tf.constant(-1).numpy()]
    name = "case3"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": default, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    branch_index = tf.constant(0)
    branch_fns = [lambda: tf.constant([[1, 2], [3, 4]]).numpy(), lambda: tf.constant([[5, 6], [7, 8]]).numpy()]
    default = [lambda: tf.constant([[-1, -2], [-3, -4]]).numpy()]
    name = "case4"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": default, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    branch_index = tf.constant(1)
    branch_fns = [lambda: tf.constant(1.0).numpy(), lambda: tf.constant(2.0).numpy()]
    default = [lambda: tf.constant(-1.0).numpy()]
    name = "case5"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": default, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    branch_index = tf.constant(0)
    branch_fns = [lambda: tf.constant([1, 2, 3]).numpy(), lambda: tf.constant([4, 5, 6]).numpy()]
    default = [lambda: tf.constant([-1, -2, -3]).numpy()]
    name = "case6"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": default, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    branch_index = tf.constant(2)
    branch_fns = [lambda: tf.constant(1).numpy(), lambda: tf.constant(2).numpy(), lambda: tf.constant(3).numpy()]
    default = [lambda: tf.constant(-1).numpy()]
    name = "case7"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": default, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    branch_index = tf.constant(0)
    branch_fns = [lambda: tf.constant(1).numpy(), lambda: tf.constant(2).numpy(), lambda: tf.constant(3).numpy()]
    default = [lambda: tf.constant(-1).numpy()]
    name = "case8"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": default, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    branch_index = tf.constant(2)
    branch_fns = [lambda: tf.constant([1, 2, 3]).numpy(), lambda: tf.constant([4, 5, 6]).numpy(), lambda: tf.constant([7, 8, 9]).numpy()]
    default = [lambda: tf.constant([-1, -2, -3]).numpy()]
    name = "case9"
    input_dict = {"branch_index": branch_index, "branch_fns": branch_fns, "default": default, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    branch_index = tf.constant(1)
    branch_fns = [lambda: tf.constant([[1, 2], [3, 4]]).numpy(), lambda: tf.constant([[5, 6], [7, 8]]).numpy()]
    default = [lambda: tf.constant([[-1, -2], [-3, -4]]).numpy()]
    name = "case10"
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
