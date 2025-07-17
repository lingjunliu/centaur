
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_StringSplitV2_inputs():
    list_of_inputs = []

    # Input 1: Basic example with space as separator
    input_val = np.array(["hello world", "a b c"], dtype=np.string_)
    sep_val = np.array(" ", dtype=np.string_)
    maxsplit_val = -1
    name_val = None
    input_dict = {"input": input_val, "sep": sep_val, "maxsplit": maxsplit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Using a different separator
    input_val = np.array(["1,2,3", "4,5,6"], dtype=np.string_)
    sep_val = np.array(",", dtype=np.string_)
    maxsplit_val = -1
    name_val = None
    input_dict = {"input": input_val, "sep": sep_val, "maxsplit": maxsplit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Limiting the number of splits
    input_val = np.array(["one two three four"], dtype=np.string_)
    sep_val = np.array(" ", dtype=np.string_)
    maxsplit_val = 2
    name_val = None
    input_dict = {"input": input_val, "sep": sep_val, "maxsplit": maxsplit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty strings
    input_val = np.array([""], dtype=np.string_)
    sep_val = np.array(" ", dtype=np.string_)
    maxsplit_val = -1
    name_val = None
    input_dict = {"input": input_val, "sep": sep_val, "maxsplit": maxsplit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple separators in a row
    input_val = np.array(["1<>2<><>3"], dtype=np.string_)
    sep_val = np.array("<>", dtype=np.string_)
    maxsplit_val = -1
    name_val = None
    input_dict = {"input": input_val, "sep": sep_val, "maxsplit": maxsplit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: No separator provided (whitespace separation)
    input_val = np.array([" hello  world  "], dtype=np.string_)
    sep_val = np.array("", dtype=np.string_)
    maxsplit_val = -1
    name_val = None
    input_dict = {"input": input_val, "sep": sep_val, "maxsplit": maxsplit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different separator characters
    input_val = np.array(["a-b-c-d"], dtype=np.string_)
    sep_val = np.array("-", dtype=np.string_)
    maxsplit_val = -1
    name_val = None
    input_dict = {"input": input_val, "sep": sep_val, "maxsplit": maxsplit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  maxsplit=0
    input_val = np.array(["one two three"], dtype=np.string_)
    sep_val = np.array(" ", dtype=np.string_)
    maxsplit_val = 0
    name_val = None
    input_dict = {"input": input_val, "sep": sep_val, "maxsplit": maxsplit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: A more complex example
    input_val = np.array(["this is,a test,string"], dtype=np.string_)
    sep_val = np.array(",", dtype=np.string_)
    maxsplit_val = 1
    name_val = None
    input_dict = {"input": input_val, "sep": sep_val, "maxsplit": maxsplit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty separator with leading/trailing whitespace
    input_val = np.array(["  leading and trailing  "], dtype=np.string_)
    sep_val = np.array("", dtype=np.string_)
    maxsplit_val = -1
    name_val = None
    input_dict = {"input": input_val, "sep": sep_val, "maxsplit": maxsplit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Empty separator with no leading/trailing whitespace
    input_val = np.array(["onetwothree"], dtype=np.string_)
    sep_val = np.array("", dtype=np.string_)
    maxsplit_val = -1
    name_val = None
    input_dict = {"input": input_val, "sep": sep_val, "maxsplit": maxsplit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Longer strings
    input_val = np.array(["This is a very long string to test the split function."], dtype=np.string_)
    sep_val = np.array(" ", dtype=np.string_)
    maxsplit_val = -1
    name_val = None
    input_dict = {"input": input_val, "sep": sep_val, "maxsplit": maxsplit_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13: maxsplit > number of splits
    input_val = np.array(["a b"], dtype=np.string_)
    sep_val = np.array(" ", dtype=np.string_)
    maxsplit_val = 5
    name_val = None
    input_dict = {"input": input_val, "sep": sep_val, "maxsplit": maxsplit_val, "name": name_val}
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
