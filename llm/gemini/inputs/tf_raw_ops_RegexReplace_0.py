
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_regex_replace_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array("hello world", dtype=np.object_)
    pattern_tensor = np.array("world", dtype=np.object_)
    rewrite_tensor = np.array("tensorflow", dtype=np.object_)
    replace_global_val = True
    name_val = "replace1"
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array("hello hello world", dtype=np.object_)
    pattern_tensor = np.array("hello", dtype=np.object_)
    rewrite_tensor = np.array("tensorflow", dtype=np.object_)
    replace_global_val = True
    name_val = "replace2"
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array("hello hello world", dtype=np.object_)
    pattern_tensor = np.array("hello", dtype=np.object_)
    rewrite_tensor = np.array("tensorflow", dtype=np.object_)
    replace_global_val = False
    name_val = "replace3"
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array(["hello world", "goodbye world"], dtype=np.object_)
    pattern_tensor = np.array("world", dtype=np.object_)
    rewrite_tensor = np.array("tensorflow", dtype=np.object_)
    replace_global_val = True
    name_val = "replace4"
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array("hello world", dtype=np.object_)
    pattern_tensor = np.array("w.*d", dtype=np.object_)
    rewrite_tensor = np.array("tensorflow", dtype=np.object_)
    replace_global_val = True
    name_val = "replace5"
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array("hello world hello", dtype=np.object_)
    pattern_tensor = np.array("hello", dtype=np.object_)
    rewrite_tensor = np.array("", dtype=np.object_)
    replace_global_val = True
    name_val = "replace6"
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array("123 abc 456", dtype=np.object_)
    pattern_tensor = np.array("[0-9]+", dtype=np.object_)
    rewrite_tensor = np.array("X", dtype=np.object_)
    replace_global_val = True
    name_val = "replace7"
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array("hello world 123 goodbye", dtype=np.object_)
    pattern_tensor = np.array("[a-z]+", dtype=np.object_)
    rewrite_tensor = np.array("Y", dtype=np.object_)
    replace_global_val = False
    name_val = "replace8"
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array("hello world", dtype=np.object_)
    pattern_tensor = np.array("^hello", dtype=np.object_)
    rewrite_tensor = np.array("hi", dtype=np.object_)
    replace_global_val = True
    name_val = "replace9"
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array("hello world\nnew line", dtype=np.object_)
    pattern_tensor = np.array(".*", dtype=np.object_)
    rewrite_tensor = np.array("replaced", dtype=np.object_)
    replace_global_val = True
    name_val = "replace10"
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RegexReplace"] = tf_raw_ops_regex_replace_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RegexReplace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RegexReplace'.")

check_valid('tf.raw_ops.RegexReplace', generated_inputs['tf.raw_ops.RegexReplace'], lib="tf", suffix=0)
