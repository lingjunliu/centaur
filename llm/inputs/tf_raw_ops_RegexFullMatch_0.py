
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_raw_ops_regex_full_match_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.RegexFullMatch operation.
    """
    list_of_inputs = []

    # Input 1: Basic 1D match
    input_dict = {
        'input': np.array([b"apple", b"banana", b"apricot"], dtype=object),
        'pattern': np.array(b"ap.*", dtype=object),
        'name': 'basic_1d_match'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D match at the end of the string
    input_dict = {
        'input': np.array([b"TF lib", b"lib TF"], dtype=object),
        'pattern': np.array(b".*TF$", dtype=object),
        'name': 'end_of_string_match'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D input tensor
    input_dict = {
        'input': np.array([[b"cat", b"dog"], [b"bat", b"rat"]], dtype=object),
        'pattern': np.array(b".at", dtype=object),
        'name': '2d_input_match'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Matching digits
    input_dict = {
        'input': np.array([b"Product_123", b"Item-456", b"Service789", b"NoNumber"], dtype=object),
        'pattern': np.array(b".*\\d{3}", dtype=object),
        'name': 'digit_match'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Matching only empty strings
    input_dict = {
        'input': np.array([b"a", b"", b"b", b""], dtype=object),
        'pattern': np.array(b"^$", dtype=object),
        'name': 'empty_string_match'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Pattern with character classes
    input_dict = {
        'input': np.array([b"Gray", b"grey", b"GReY"], dtype=object),
        'pattern': np.array(b"Gr[ae]y", dtype=object),
        'name': 'char_class_match'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: No matches
    input_dict = {
        'input': np.array([b"abc", b"def", b"ghi"], dtype=object),
        'pattern': np.array(b"xyz", dtype=object),
        'name': 'no_match_case'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Universal match
    input_dict = {
        'input': np.array([b"any string", b"123!@#", b"", b"\n\t"], dtype=object),
        'pattern': np.array(b".*", dtype=object),
        'name': 'universal_match'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Matching special regex characters by escaping them
    input_dict = {
        'input': np.array([b"(abc)", b"[def]", b"{ghi}"], dtype=object),
        'pattern': np.array(b"\\(abc\\)", dtype=object),
        'name': 'special_char_escape'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D input tensor
    input_dict = {
        'input': np.array([[[b"a"], [b"b"]], [[b"ab"], [b"ba"]]], dtype=object),
        'pattern': np.array(b"a.*", dtype=object),
        'name': '3d_input_match'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Empty input tensor
    input_dict = {
        'input': np.array([], dtype=object),
        'pattern': np.array(b".*", dtype=object),
        'name': 'empty_input_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Scalar input tensor
    input_dict = {
        'input': np.array(b"just_one_string", dtype=object),
        'pattern': np.array(b"just_one_string", dtype=object),
        'name': 'scalar_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.RegexFullMatch"] = get_tf_raw_ops_regex_full_match_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RegexFullMatch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RegexFullMatch'.")

check_valid('tf.raw_ops.RegexFullMatch', generated_inputs['tf.raw_ops.RegexFullMatch'], lib="tf", suffix=0)
