
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_strings_lower_inputs():
    """
    Generates a list of valid inputs for the tf.strings.lower function.
    """
    list_of_inputs = []

    # Input 1: Basic scalar string
    input_dict_1 = {
        'input': np.array(b'HELLO WORLD', dtype=object),
        'encoding': '',
        'name': 'test1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D Tensor of strings with mixed casing
    input_dict_2 = {
        'input': np.array([b'CamelCase', b'ALLCAPS', b'lowercase', b'with 123 NUMBERS'], dtype=object),
        'encoding': '',
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D Tensor of strings
    input_dict_3 = {
        'input': np.array([[b'FIRST ROW', b'FIRST-ROW-COL2'], [b'Second row', b'SECOND-ROW-COL2']], dtype=object),
        'encoding': '',
        'name': 'test3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Scalar empty string
    input_dict_4 = {
        'input': np.array(b'', dtype=object),
        'encoding': '',
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Tensor containing an empty string
    input_dict_5 = {
        'input': np.array([b'A', b'', b'B', b''], dtype=object),
        'encoding': '',
        'name': 'test5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: UTF-8 encoding with non-ASCII characters
    input_dict_6 = {
        'input': np.array(['MÜNCHEN'.encode('utf-8'), 'STRASSE'.encode('utf-8'), 'ÇA VA?'.encode('utf-8')], dtype=object),
        'encoding': 'utf-8',
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Scalar UTF-8 string
    input_dict_7 = {
        'input': np.array('ΔΘΠ'.encode('utf-8'), dtype=object),
        'encoding': 'utf-8',
        'name': 'test7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Empty tensor (shape (0,))
    input_dict_8 = {
        'input': np.array([], dtype=object),
        'encoding': '',
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: 3D Tensor of strings
    input_dict_9 = {
        'input': np.array([[[b'A'], [b'B']], [[b'C'], [b'd']]], dtype=object),
        'encoding': '',
        'name': 'test9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Strings that are already lowercase
    input_dict_10 = {
        'input': np.array([b'already', b'all', b'lower'], dtype=object),
        'encoding': '',
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Strings with only symbols and numbers
    input_dict_11 = {
        'input': np.array([b'12345', b'!@#$%^&*()', b'1.2.3-RELEASE'], dtype=object),
        'encoding': '',
        'name': 'test11'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    return list_of_inputs

generated_inputs["tf.strings.lower"] = get_tf_strings_lower_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.strings.lower' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.lower'.")

check_valid('tf.strings.lower', generated_inputs['tf.strings.lower'], lib="tf", suffix=0)
