
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_stringstrip_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.StringStrip operation.
    """
    list_of_inputs = []

    # Input 1: Basic 1D array with mixed whitespace
    input_dict = {
        'input': np.array(["  hello", "world  ", "  both  ", "none"], dtype=object),
        'name': 'basic_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array with various whitespace characters (tabs, newlines)
    input_dict = {
        'input': np.array(["\tTensorFlow", "The python library\n\n", "\n\t   mixed   \t\n"], dtype=object),
        'name': 'special_whitespace_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array
    input_dict = {
        'input': np.array([[" leading", "trailing ", "  both  "], ["\nnewline", "tab\t", " no_space "]], dtype=object),
        'name': 'basic_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Edge cases with empty and whitespace-only strings
    input_dict = {
        'input': np.array(["", " ", "   ", "\t", "\n", " non-empty "], dtype=object),
        'name': 'edge_cases_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar (0D tensor)
    input_dict = {
        'input': np.array("  scalar string  ", dtype=object),
        'name': 'scalar_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array
    input_dict = {
        'input': np.array([[[" a ", "b"], [" c", "d "]], [["\te\t", "\nf\n"], [" g ", "h"]]], dtype=object),
        'name': 'basic_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Already stripped strings (no change expected)
    input_dict = {
        'input': np.array(["already", "stripped", "strings"], dtype=object),
        'name': 'already_stripped'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Unicode characters with whitespace
    input_dict = {
        'input': np.array(["  你好  ", "こんにちは  ", "  привет"], dtype=object),
        'name': 'unicode_strings'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array mixing various cases
    input_dict = {
        'input': np.array([["", "  a  ", "b"], [" c ", " ", "\td\t"]], dtype=object),
        'name': 'mixed_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Scalar whitespace-only string
    input_dict = {
        'input': np.array("\t  \n", dtype=object),
        'name': 'scalar_whitespace_only'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: An empty 1D tensor
    input_dict = {
        'input': np.array([], dtype=object),
        'name': 'empty_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: A 2D tensor with an empty dimension
    input_dict = {
        'input': np.empty(shape=(2,0), dtype=object),
        'name': 'empty_dimension_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs["tf.raw_ops.StringStrip"] = tf_raw_ops_stringstrip_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.StringStrip' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringStrip'.")

check_valid('tf.raw_ops.StringStrip', generated_inputs['tf.raw_ops.StringStrip'], lib="tf", suffix=0)
