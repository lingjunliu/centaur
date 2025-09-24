
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import collections

def get_tf_nest_assert_same_structure_inputs():
    """
    Generates a list of valid inputs for the tf.nest.assert_same_structure function.
    All lists are homogeneous to be compatible with external validation scripts that
    may attempt to convert them to a single NumPy array.
    """
    list_of_inputs = []

    # Input 1: Simple list of scalar arrays.
    input_dict_1 = {
        'nest1': [np.array(1), np.array(2), np.array(3)],
        'nest2': [np.array(4), np.array(5), np.array(6)],
        'check_types': True,
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Simple list of 1D arrays of the same shape.
    input_dict_2 = {
        'nest1': [np.array([1, 2]), np.array([3, 4])],
        'nest2': [np.array([5, 6]), np.array([7, 8])],
        'check_types': True,
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Simple list of 2D arrays of the same shape.
    input_dict_3 = {
        'nest1': [np.array([[1]]), np.array([[2]])],
        'nest2': [np.array([[3]]), np.array([[4]])],
        'check_types': True,
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Nested list of scalar arrays, homogeneous at each level.
    input_dict_4 = {
        'nest1': [[np.array(1), np.array(2)], [np.array(3), np.array(4)]],
        'nest2': [[np.array(5), np.array(6)], [np.array(7), np.array(8)]],
        'check_types': True,
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Nested list of 1D arrays, homogeneous at each level.
    input_dict_5 = {
        'nest1': [[np.array([1, 2])], [np.array([3, 4])]],
        'nest2': [[np.array([5, 6])], [np.array([7, 8])]],
        'check_types': True,
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Top-level empty lists.
    input_dict_6 = {
        'nest1': [],
        'nest2': [],
        'check_types': True,
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: List with negative float values.
    input_dict_7 = {
        'nest1': [np.array(-1.5), np.array(-2.5)],
        'nest2': [np.array(-10.0), np.array(-20.0)],
        'check_types': True,
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Deeper nesting, homogeneous.
    input_dict_8 = {
        'nest1': [[[np.array(1)], [np.array(2)]], [[np.array(3)], [np.array(4)]]],
        'nest2': [[[np.array(5)], [np.array(6)]], [[np.array(7)], [np.array(8)]]],
        'check_types': True,
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: check_types=False, list of tuples vs list of lists
    input_dict_9 = {
        'nest1': [(np.array(1),), (np.array(2),)],
        'nest2': [[np.array(3)], [np.array(4)]],
        'check_types': False,
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: check_types=False, deeper structure.
    input_dict_10 = {
        'nest1': [[(np.array(1), np.array(2))]],
        'nest2': [[[np.array(3), np.array(4)]]],
        'check_types': False,
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.nest.assert_same_structure"] = get_tf_nest_assert_same_structure_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nest.assert_same_structure' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nest.assert_same_structure'.")

check_valid('tf.nest.assert_same_structure', generated_inputs['tf.nest.assert_same_structure'], lib="tf", suffix=0)
