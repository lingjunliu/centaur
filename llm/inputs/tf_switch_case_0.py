
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# Define a set of simple callable functions that return TF constants
def f_int_1(): return tf.constant(1, dtype=tf.int32)
def f_int_2(): return tf.constant(2, dtype=tf.int32)
def f_int_default(): return tf.constant(-1, dtype=tf.int32)

def f_vec_1(): return tf.constant([1, 2], dtype=tf.int32)
def f_vec_2(): return tf.constant([3, 4], dtype=tf.int32)
def f_vec_default(): return tf.constant([9, 9], dtype=tf.int32)

# This input generation circumvents a testing framework error by using a dictionary for 'branch_fns'
# and a single callable for 'default', as permitted by the TensorFlow API documentation.
# This avoids passing a list of functions, which causes a TypeError in the test environment.
def tf_switch_case_inputs():
    list_of_inputs = []

    # Input 1: Basic case, select index 0. `branch_fns` is a dict.
    input_dict_1 = {
        'branch_index': np.array(0, dtype=np.int32),
        'branch_fns': {0: f_int_1, 1: f_int_2},
        'default': f_int_default,
        'name': 'select_0_dict'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Select index 1.
    input_dict_2 = {
        'branch_index': np.array(1, dtype=np.int32),
        'branch_fns': {0: f_vec_1, 1: f_vec_2},
        'default': f_vec_default,
        'name': 'select_1_vec_dict'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Fallback to default, index out of bounds.
    input_dict_3 = {
        'branch_index': np.array(5, dtype=np.int32),
        'branch_fns': {0: f_int_1, 1: f_int_2},
        'default': f_int_default,
        'name': 'default_oob_dict'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: No default provided. Falls back to max-keyed branch fn.
    input_dict_4 = {
        'branch_index': np.array(10, dtype=np.int32),
        'branch_fns': {0: f_vec_1, 1: f_vec_2},
        'default': None,
        'name': 'no_default_fallback_dict'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Single branch, selected.
    input_dict_5 = {
        'branch_index': np.array(0, dtype=np.int32),
        'branch_fns': {0: f_vec_1},
        'default': f_vec_default,
        'name': 'single_branch_selected_dict'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Single branch, not selected (use default).
    input_dict_6 = {
        'branch_index': np.array(1, dtype=np.int32),
        'branch_fns': {0: f_vec_1},
        'default': f_vec_default,
        'name': 'single_branch_default_dict'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    return list_of_inputs

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

check_valid('tf.switch_case', generated_inputs['tf.switch_case'], lib="tf", suffix=0)
