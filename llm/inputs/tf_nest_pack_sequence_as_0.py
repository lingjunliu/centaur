
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

    # Input 4: Nested dictionary and tuple
    structure = {"a": (1, 2), "b": {"c": 3, "d": 4}}
    flat_sequence = [5, 6, 7, 8]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Deeper nesting.  Simplified structure
    structure = [((1, 2), 3), 4]
    flat_sequence = [7, 8, 9, 10]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty structure
    structure = []
    flat_sequence = []
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: List of integers
    structure = [1, 2]
    flat_sequence = [2, 3]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex structure.  Simplified
    structure = {"a": 1, "c": 4}
    flat_sequence = [6, 8]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Boolean Values
    structure = [True, False]
    flat_sequence = [False, True]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Expand Composites with a simple structure
    structure = [1, 2]
    flat_sequence = [4, 5]
    expand_composites = True
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
