
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nest_assert_same_structure_inputs():
    list_of_inputs = []

    # Input 1
    nest1 = [1, 2, 3]
    nest2 = [4, 5, 6]
    check_types = True
    expand_composites = False
    input_dict = {"nest1": nest1, "nest2": nest2, "check_types": check_types, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    nest1 = [[1, 2], [3, 4]]
    nest2 = [[5, 6], [7, 8]]
    check_types = False
    expand_composites = True
    input_dict = {"nest1": nest1, "nest2": nest2, "check_types": check_types, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    nest1 = [1, [2, 3], 4]
    nest2 = [5, [6, 7], 8]
    check_types = True
    expand_composites = False
    input_dict = {"nest1": nest1, "nest2": nest2, "check_types": check_types, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    nest1 = [[1, 2], [3, 4]]
    nest2 = [[6, 7], [8, 9]]
    check_types = False
    expand_composites = False
    input_dict = {"nest1": nest1, "nest2": nest2, "check_types": check_types, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    nest1 = [1, 2, [3, 4]]
    nest2 = [6, 7, [8, 9]]
    check_types = True
    expand_composites = False
    input_dict = {"nest1": nest1, "nest2": nest2, "check_types": check_types, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    nest1 = [[1, 2], 3, [4, 5]]
    nest2 = [[7, 8], 9, [10, 11]]
    check_types = False
    expand_composites = False
    input_dict = {"nest1": nest1, "nest2": nest2, "check_types": check_types, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    nest1 = [[1,2], [3,4]]
    nest2 = [[3,4], [5,6]]
    check_types = True
    expand_composites = False
    input_dict = {"nest1": nest1, "nest2": nest2, "check_types": check_types, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    nest1 = [[[1]], [[2]]]
    nest2 = [[[3]], [[4]]]
    check_types = False
    expand_composites = False
    input_dict = {"nest1": nest1, "nest2": nest2, "check_types": check_types, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nest.assert_same_structure"] = tf_nest_assert_same_structure_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nest.assert_same_structure' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nest.assert_same_structure'.")

check_valid('tf.nest.assert_same_structure', generated_inputs['tf.nest.assert_same_structure'], lib="tf", suffix=0)
