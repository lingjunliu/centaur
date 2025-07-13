
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_StringSplitV2_inputs():
    list_of_inputs = []

    # Input 1: Basic case with space as separator
    input_tensor = np.array(["hello world", "a b c"], dtype=np.string_)
    sep_tensor = np.array(" ", dtype=np.string_)
    maxsplit = -1
    name = None

    input_dict = {
        "input": input_tensor,
        "sep": sep_tensor,
        "maxsplit": maxsplit,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different separator
    input_tensor = np.array(["1,2,3", "4,5,6"], dtype=np.string_)
    sep_tensor = np.array(",", dtype=np.string_)
    maxsplit = -1
    name = None

    input_dict = {
        "input": input_tensor,
        "sep": sep_tensor,
        "maxsplit": maxsplit,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Maxsplit > 0
    input_tensor = np.array(["a b c d"], dtype=np.string_)
    sep_tensor = np.array(" ", dtype=np.string_)
    maxsplit = 2
    name = None

    input_dict = {
        "input": input_tensor,
        "sep": sep_tensor,
        "maxsplit": maxsplit,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty separator
    input_tensor = np.array(["hello world"], dtype=np.string_)
    sep_tensor = np.array("", dtype=np.string_)
    maxsplit = -1
    name = None

    input_dict = {
        "input": input_tensor,
        "sep": sep_tensor,
        "maxsplit": maxsplit,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Consecutive delimiters
    input_tensor = np.array(["1<>2<><>3"], dtype=np.string_)
    sep_tensor = np.array("<>", dtype=np.string_)
    maxsplit = -1
    name = None

    input_dict = {
        "input": input_tensor,
        "sep": sep_tensor,
        "maxsplit": maxsplit,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Input with leading/trailing whitespace, empty separator
    input_tensor = np.array(["  hello world  "], dtype=np.string_)
    sep_tensor = np.array("", dtype=np.string_)
    maxsplit = -1
    name = None

    input_dict = {
        "input": input_tensor,
        "sep": sep_tensor,
        "maxsplit": maxsplit,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Input with multiple spaces, empty separator
    input_tensor = np.array(["hello   world"], dtype=np.string_)
    sep_tensor = np.array("", dtype=np.string_)
    maxsplit = -1
    name = None

    input_dict = {
        "input": input_tensor,
        "sep": sep_tensor,
        "maxsplit": maxsplit,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Maxsplit 0
    input_tensor = np.array(["a b c d"], dtype=np.string_)
    sep_tensor = np.array(" ", dtype=np.string_)
    maxsplit = 0
    name = None

    input_dict = {
        "input": input_tensor,
        "sep": sep_tensor,
        "maxsplit": maxsplit,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Sep is a longer string
    input_tensor = np.array(["applebananaorange"], dtype=np.string_)
    sep_tensor = np.array("banana", dtype=np.string_)
    maxsplit = -1
    name = None

    input_dict = {
        "input": input_tensor,
        "sep": sep_tensor,
        "maxsplit": maxsplit,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty input string
    input_tensor = np.array([""], dtype=np.string_)
    sep_tensor = np.array(" ", dtype=np.string_)
    maxsplit = -1
    name = None

    input_dict = {
        "input": input_tensor,
        "sep": sep_tensor,
        "maxsplit": maxsplit,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Test with different maxsplit values
    input_tensor = np.array(["one two three four"], dtype=np.string_)
    sep_tensor = np.array(" ", dtype=np.string_)
    maxsplit = 1
    name = None

    input_dict = {
        "input": input_tensor,
        "sep": sep_tensor,
        "maxsplit": maxsplit,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StringSplitV2"] = tf_raw_ops_StringSplitV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StringSplitV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringSplitV2'.")

check_valid('tf.raw_ops.StringSplitV2', generated_inputs['tf.raw_ops.StringSplitV2'], lib="tf", suffix=0)
