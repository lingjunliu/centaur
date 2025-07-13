
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
    structure = {"a": 1, "b": 2, "c": 3}
    flat_sequence = [4, 5, 6]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Nested dictionary
    structure = {"a": {"b": 1, "c": 2}, "d": 3}
    flat_sequence = [4, 5, 6]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List with a numpy array
    structure = [np.array([1])]
    flat_sequence = [np.array([4])]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: More complex nested structure
    structure = ({"a": 1, "b": (2,)}, [4, 5])
    flat_sequence = [6, 7, 8, 9, 10]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: Empty structure
    structure = []
    flat_sequence = []
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single element structure
    structure = [1]
    flat_sequence = [2]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Structure with numpy array and different data types
    structure = [np.array([1.0]), 3]
    flat_sequence = [np.array([4.0]), 6]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tuple with a numpy array.

    structure = (1,np.array(2))
    flat_sequence = [3,np.array(4)]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Boolean values

    structure = [True, False]
    flat_sequence = [False, True]
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
