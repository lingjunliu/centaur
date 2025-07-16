
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nest_pack_sequence_as_inputs():
    list_of_inputs = []

    # Input 1: Simple list structure
    structure = [1, 2, 3]
    flat_sequence = [4, 5, 6]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Nested tuple structure
    structure = ((1, 2), (3, 4, 5))
    flat_sequence = [6, 7, 8, 9, 10]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Dictionary structure
    structure = {"a": 1, "b": 2}
    flat_sequence = [3, 4]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Nested dictionary and tuple structure
    structure = {"a": (1, 2), "b": {"c": 3, "d": 4}}
    flat_sequence = [5, 6, 7, 8]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List with same data types to avoid ValueError later
    structure = [1.0, 2.0, 3.0]
    flat_sequence = [4.0, 5.0, 6.0]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Deeper nested structure
    structure = [[1, [2, 3]], 4]
    flat_sequence = [5, 6, 7, 8]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty structure
    structure = []
    flat_sequence = []
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: List with numpy arrays - Using single element arrays consistently
    structure = [np.array([1]), np.array([3])]
    flat_sequence = [np.array([4]), np.array([5])]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Dictionary with numpy arrays - Using single element arrays consistently
    structure = {"a": np.array([1]), "b": np.array([2])}
    flat_sequence = [np.array([4]), np.array([6])]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Nested structure with numpy arrays - Using single element arrays consistently and same type, and consistent structure for numbers
    structure = {"a": [np.array([1])], "b": (np.array([4]), np.array([5]))}
    flat_sequence = [np.array([6]), np.array([7]), np.array([8])]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: More numpy array tests, consistent shape
    structure = [np.array([1, 2]), np.array([3, 4])]
    flat_sequence = [np.array([5, 6]), np.array([7, 8])]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Float numpy arrays
    structure = [np.array([1.0]), np.array([2.0])]
    flat_sequence = [np.array([3.0]), np.array([4.0])]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13: More complex numpy structure - Ensure consistency
    structure = {"a": (np.array([1, 2]), np.array([3])), "b": [np.array([4]), np.array([5])]}
    flat_sequence = [np.array([6, 7]), np.array([8]), np.array([9]), np.array([10])]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14: Structure only with numpy arrays of consistent types and shapes
    structure = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    flat_sequence = [np.array([5.0, 6.0]), np.array([7.0, 8.0])]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nest.pack_sequence_as"] = tf_nest_pack_sequence_as_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nest.pack_sequence_as' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nest.pack_sequence_as'.")

check_valid('tf.nest.pack_sequence_as', generated_inputs['tf.nest.pack_sequence_as'], lib="tf", suffix=0)
