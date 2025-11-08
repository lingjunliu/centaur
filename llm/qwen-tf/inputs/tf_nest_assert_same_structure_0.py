
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nest_assert_same_structure_inputs():
    list_of_inputs = []
    
    # Input 1 - Valid: Same nested structure with same types
    nest1 = [[1, 2], [3, 4]]
    nest2 = [[5, 6], [7, 8]]
    check_types = True
    expand_composite = False
    
    input_dict = {
        "nest1": nest1,
        "nest2": nest2,
        "check_types": check_types,
        "expand_composite": expand_composite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - Valid: Same nested structure with different types (check_types=False)
    nest1 = [1, 2]
    nest2 = (3, 4)
    check_types = False
    expand_composite = False
    
    input_dict = {
        "nest1": nest1,
        "nest2": nest2,
        "check_types": check_types,
        "expand_composite": expand_composite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - Valid: Same nested structure with different types (check_types=False)
    nest1 = {"a": 1, "b": 2}
    nest2 = {"c": 3, "d": 4}
    check_types = False
    expand_composite = False
    
    input_dict = {
        "nest1": nest1,
        "nest2": nest2,
        "check_types": check_types,
        "expand_composite": expand_composite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - Valid: Same nested structure with same types (check_types=True)
    nest1 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    nest2 = [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
    check_types = True
    expand_composite = False
    
    input_dict = {
        "nest1": nest1,
        "nest2": nest2,
        "check_types": check_types,
        "expand_composite": expand_composite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - Valid: Same nested structure with different types (check_types=False)
    nest1 = [1, 2, 3]
    nest2 = [4, 5, 6]
    check_types = False
    expand_composite = False
    
    input_dict = {
        "nest1": nest1,
        "nest2": nest2,
        "check_types": check_types,
        "expand_composite": expand_composite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - Valid: Same nested structure with same types (check_types=True)
    nest1 = np.array([[1, 2], [3, 4]])
    nest2 = np.array([[5, 6], [7, 8]])
    check_types = True
    expand_composite = False
    
    input_dict = {
        "nest1": nest1,
        "nest2": nest2,
        "check_types": check_types,
        "expand_composite": expand_composite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Valid: Same nested structure with different types (check_types=False)
    nest1 = np.array([1, 2, 3])
    nest2 = np.array([4, 5, 6])
    check_types = False
    expand_composite = False
    
    input_dict = {
        "nest1": nest1,
        "nest2": nest2,
        "check_types": check_types,
        "expand_composite": expand_composite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - Valid: Same nested structure with same types (check_types=True)
    nest1 = np.array([[1, 2], [3, 4]])
    nest2 = np.array([[5, 6], [7, 8]])
    check_types = True
    expand_composite = False
    
    input_dict = {
        "nest1": nest1,
        "nest2": nest2,
        "check_types": check_types,
        "expand_composite": expand_composite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Valid: Same nested structure with different types (check_types=False)
    nest1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    nest2 = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    check_types = False
    expand_composite = False
    
    input_dict = {
        "nest1": nest1,
        "nest2": nest2,
        "check_types": check_types,
        "expand_composite": expand_composite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - Valid: Same nested structure with same types (check_types=True)
    nest1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    nest2 = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    check_types = True
    expand_composite = False
    
    input_dict = {
        "nest1": nest1,
        "nest2": nest2,
        "check_types": check_types,
        "expand_composite": expand_composite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nest.assert_same_structure"] = tf_nest_assert_same_structure_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nest.assert_same_structure' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nest.assert_same_structure'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nest.assert_same_structure', generated_inputs['tf.nest.assert_same_structure'], lib="tf", suffix=0)
