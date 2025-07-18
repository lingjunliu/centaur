
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_nest_pack_sequence_as_inputs():
    """
    Generates a list of valid inputs for tf.nest.pack_sequence_as.
    The inputs are constrained to be homogeneous lists to avoid errors
    with testing frameworks that might apply np.min/np.max on them.
    """
    list_of_inputs = []

    # Input 1: Simple flat list structure with floats.
    input_dict_1 = {
        'structure': [0, 1, 2, 3],
        'flat_sequence': [1.0, 2.0, 3.0, 4.0],
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Simple 2D, non-ragged list structure with negative integers.
    input_dict_2 = {
        'structure': [[0, 1], [2, 3]],
        'flat_sequence': [-1, -2, -3, -4],
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Column-vector-like list structure (non-ragged).
    input_dict_3 = {
        'structure': [[0], [1], [2]],
        'flat_sequence': [100, 200, 300],
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Row-vector-like list structure with expand_composites=True.
    input_dict_4 = {
        'structure': [[0, 1, 2]],
        'flat_sequence': [5, 6, 7],
        'expand_composites': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Structure with a single atom.
    input_dict_5 = {
        'structure': [99],
        'flat_sequence': [123.45],
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty structure and empty sequence.
    input_dict_6 = {
        'structure': [],
        'flat_sequence': [],
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Deeper but regular (non-ragged) nesting.
    input_dict_7 = {
        'structure': [[[0, 1]], [[2, 3]]],
        'flat_sequence': [0.1, 0.2, 0.3, 0.4],
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: flat_sequence with mixed numeric numpy types.
    input_dict_8 = {
        'structure': [0, 1, 2],
        'flat_sequence': [np.int16(10), np.float32(20.5), np.int64(30)],
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Flat structure, longer sequence.
    input_dict_9 = {
        'structure': [0, 1, 2, 3, 4, 5, 6, 7],
        'flat_sequence': [0, -1, 2, -3, 4, -5, 6, -7],
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 3D regular structure.
    input_dict_10 = {
        'structure': [[[[0], [1]], [[2], [3]]], [[[4], [5]], [[6], [7]]]],
        'flat_sequence': [1, 2, 3, 4, 5, 6, 7, 8],
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    # Input 11: Single numpy array in flat sequence
    input_dict_11 = {
        'structure': [0],
        'flat_sequence': [np.array([[1, 2], [3, 4]], dtype=np.int32)],
        'expand_composites': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    return list_of_inputs

generated_inputs["tf.nest.pack_sequence_as"] = tf_nest_pack_sequence_as_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nest.pack_sequence_as' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nest.pack_sequence_as'.")

check_valid('tf.nest.pack_sequence_as', generated_inputs['tf.nest.pack_sequence_as'], lib="tf", suffix=0)
