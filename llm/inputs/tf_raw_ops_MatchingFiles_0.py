
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os

def tf_raw_ops_matchingfiles_inputs():
    list_of_inputs = []

    # Input 1: Basic pattern
    pattern = np.array("*.txt", dtype=np.string_)
    name = None
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Vector of patterns
    pattern = np.array(["*.txt", "*.log"], dtype=np.string_)
    name = "matching_files_op"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty pattern
    pattern = np.array("", dtype=np.string_)
    name = None
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Pattern with a subdirectory (not supported, but valid input)
    pattern = np.array("subdir/*.txt", dtype=np.string_)
    name = None
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex pattern
    pattern = np.array("*[0-9].txt", dtype=np.string_)
    name = "complex_pattern"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multiple wildcard characters
    pattern = np.array("file*.*", dtype=np.string_)
    name = None
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  More complex pattern
    pattern = np.array("???.txt", dtype=np.string_)
    name = None
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Pattern with character classes
    pattern = np.array("[abc]*.txt", dtype=np.string_)
    name = "char_class_pattern"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Another vector of patterns
    pattern = np.array(["a*.txt", "b*.log", "c*.py"], dtype=np.string_)
    name = "multiple_patterns"
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Pattern that should match nothing
    pattern = np.array("nonexistent*.file", dtype=np.string_)
    name = None
    input_dict = {"pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_matchingfiles_inputs()
generated_inputs["tf.raw_ops.MatchingFiles"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.MatchingFiles"].append({"kwargs": input_dict})

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MatchingFiles' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatchingFiles'.")

check_valid('tf.raw_ops.MatchingFiles', generated_inputs['tf.raw_ops.MatchingFiles'], lib="tf", suffix=0)
