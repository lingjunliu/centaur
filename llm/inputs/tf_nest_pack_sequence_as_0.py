
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
    input_dict = {'structure': structure, 'flat_sequence': flat_sequence, 'expand_composites': expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tuple structure
    structure = (1, (2, 3), 4)
    flat_sequence = [5, 6, 7, 8]
    expand_composites = False
    input_dict = {'structure': structure, 'flat_sequence': flat_sequence, 'expand_composites': expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Mixed list and tuple structure
    structure = [1, (2, [3, 4]), 5]
    flat_sequence = [6, 7, 8, 9, 10]
    expand_composites = False
    input_dict = {'structure': structure, 'flat_sequence': flat_sequence, 'expand_composites': expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Nested lists
    structure = [[1, 2], [3, 4, 5]]
    flat_sequence = [6, 7, 8, 9, 10]
    expand_composites = False
    input_dict = {'structure': structure, 'flat_sequence': flat_sequence, 'expand_composites': expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Structure with a single element
    structure = [1]
    flat_sequence = [2]
    expand_composites = False
    input_dict = {'structure': structure, 'flat_sequence': flat_sequence, 'expand_composites': expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Deeper nesting
    structure = [[[1, 2], [3]], 4]
    flat_sequence = [5, 6, 7, 8]
    expand_composites = False
    input_dict = {'structure': structure, 'flat_sequence': flat_sequence, 'expand_composites': expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty list in structure
    structure = [1, [], 2]
    flat_sequence = [3, 4]
    expand_composites = False
    input_dict = {'structure': structure, 'flat_sequence': flat_sequence, 'expand_composites': expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: List of tuples
    structure = [(1, 2), (3, 4, 5)]
    flat_sequence = [6, 7, 8, 9, 10]
    expand_composites = False
    input_dict = {'structure': structure, 'flat_sequence': flat_sequence, 'expand_composites': expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: expand_composites=True with list structure
    structure = [1, 2, 3]
    flat_sequence = [4, 5, 6]
    expand_composites = True
    input_dict = {'structure': structure, 'flat_sequence': flat_sequence, 'expand_composites': expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: numpy arrays in structure and flat_sequence, homogeneous shape
    structure = [np.array([1, 2]), np.array([3, 4])]
    flat_sequence = [np.array([5, 6]), np.array([7, 8])]
    expand_composites = False
    input_dict = {'structure': structure, 'flat_sequence': flat_sequence, 'expand_composites': expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Different numpy array shapes but still valid (all scalars)
    structure = [np.array(1), np.array(2)]
    flat_sequence = [np.array(4), np.array(5)]
    expand_composites = False
    input_dict = {'structure': structure, 'flat_sequence': flat_sequence, 'expand_composites': expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Nested numpy arrays, same shape for each nested level
    structure = [[np.array([1,1]), np.array([2,2])], [np.array([3,3])]]
    flat_sequence = [np.array([4,4]), np.array([5,5]), np.array([6,6])]
    expand_composites = False
    input_dict = {'structure': structure, 'flat_sequence': flat_sequence, 'expand_composites': expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13: Nested numpy arrays, scalar and arrays, same size
    structure = [np.array([1]), [np.array([2]), np.array([3])]]
    flat_sequence = [np.array([4]), np.array([5]), np.array([6])]
    expand_composites = False
    input_dict = {'structure': structure, 'flat_sequence': flat_sequence, 'expand_composites': expand_composites}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14: More complex nesting with numpy arrays, fixed shapes
    structure = [np.array([1]), (np.array([2]), [np.array([4]), np.array([5])])]
    flat_sequence = [np.array([6]), np.array([7]), np.array([9]), np.array([10])]
    expand_composites = False
    input_dict = {'structure': structure, 'flat_sequence': flat_sequence, 'expand_composites': expand_composites}
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
