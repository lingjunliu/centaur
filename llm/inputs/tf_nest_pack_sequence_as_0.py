
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nest_pack_sequence_as_inputs():
    list_of_inputs = []

    # Input 1
    structure = [1, 2, 3]
    flat_sequence = [4, 5, 6]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    structure = (1, (2, 3), 4)
    flat_sequence = [5, 6, 7, 8]
    expand_composites = True
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    structure = {"a": 1, "b": 2}
    flat_sequence = [3, 4]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    structure = {"a": (1, 2), "b": [3, 4]}
    flat_sequence = [5, 6, 7, 8]
    expand_composites = True
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    structure = [[1, 2], [3, 4], [5, 6]]
    flat_sequence = [7, 8, 9, 10, 11, 12]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    structure = ({"a": 1, "b": 2}, [3, 4])
    flat_sequence = [5, 6, 7, 8]
    expand_composites = True
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Modified the structure to match the flat sequence length
    structure = [[1, 2], [3, 4]]
    flat_sequence = [5, 6, 7, 8]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    structure = {"a": 1, "b": [2, 3]}
    flat_sequence = [4, 5, 6]
    expand_composites = True
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    structure = (1, {"a": 2, "b": [3, 4]}, 5)
    flat_sequence = [6, 7, 8, 9, 10]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    structure = {"x": [1,2], "y": {"a":3, "b":4}}
    flat_sequence = [5, 6, 7, 8]
    expand_composites = True
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    structure = [{"a": 1, "b": 2}, 3]
    flat_sequence = [4, 5] # Reduced flat_sequence length to match the structure
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: A simpler case
    structure = [1, 2]
    flat_sequence = [3, 4]
    expand_composites = False
    input_dict = {"structure": structure, "flat_sequence": flat_sequence, "expand_composites": expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Different data types
    structure = [1.0, 2.5]
    flat_sequence = [3.2, 4.1]
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
