
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_regex_full_match_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array(["hello world", "regex match"], dtype=np.string_)
    pattern_tensor = np.array("^hello.*$", dtype=np.string_)
    input_dict = {"input": input_tensor, "pattern": pattern_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array(["12345", "67890"], dtype=np.string_)
    pattern_tensor = np.array("^[0-9]+$", dtype=np.string_)
    input_dict = {"input": input_tensor, "pattern": pattern_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array(["abc", "def", "ghi"], dtype=np.string_)
    pattern_tensor = np.array("^[a-z]{3}$", dtype=np.string_)
    input_dict = {"input": input_tensor, "pattern": pattern_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array(["", "not empty"], dtype=np.string_)
    pattern_tensor = np.array("^$", dtype=np.string_)
    input_dict = {"input": input_tensor, "pattern": pattern_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array(["a", "b", "c"], dtype=np.string_)
    pattern_tensor = np.array("^[abc]$", dtype=np.string_)
    input_dict = {"input": input_tensor, "pattern": pattern_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array(["one", "two", "three"], dtype=np.string_)
    pattern_tensor = np.array("^(one|two|three)$", dtype=np.string_)
    input_dict = {"input": input_tensor, "pattern": pattern_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array(["file.txt", "image.png", "document.pdf"], dtype=np.string_)
    pattern_tensor = np.array(".*\\.(txt|pdf)$", dtype=np.string_)
    input_dict = {"input": input_tensor, "pattern": pattern_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array(["123-456-7890", "987-654-3210"], dtype=np.string_)
    pattern_tensor = np.array("^[0-9]{3}-[0-9]{3}-[0-9]{4}$", dtype=np.string_)
    input_dict = {"input": input_tensor, "pattern": pattern_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RegexFullMatch"] = tf_raw_ops_regex_full_match_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RegexFullMatch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RegexFullMatch'.")

check_valid('tf.raw_ops.RegexFullMatch', generated_inputs['tf.raw_ops.RegexFullMatch'], lib="tf", suffix=0)
