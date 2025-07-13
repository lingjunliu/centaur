
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_regex_replace_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array("Text with tags.<br /><b>contains html</b>").astype(np.string_)
    pattern = np.array("<[^>]+>").astype(np.string_)
    rewrite = np.array(" ").astype(np.string_)
    replace_global = True
    name = None
    input_dict = {"input": input_tensor, "pattern": pattern, "rewrite": rewrite, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array(["Hello world", "This is a test"]).astype(np.string_)
    pattern = np.array("world").astype(np.string_)
    rewrite = np.array("universe").astype(np.string_)
    replace_global = False
    name = "replace_world"
    input_dict = {"input": input_tensor, "pattern": pattern, "rewrite": rewrite, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([["a1b", "c2d"], ["e3f", "g4h"]]).astype(np.string_)
    pattern = np.array(r"(\d)").astype(np.string_)
    rewrite = np.array(r"_\1_").astype(np.string_)
    replace_global = True
    name = None
    input_dict = {"input": input_tensor, "pattern": pattern, "rewrite": rewrite, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array("123abc456def789").astype(np.string_)
    pattern = np.array(r"\d+").astype(np.string_)
    rewrite = np.array("NUM").astype(np.string_)
    replace_global = True
    name = "replace_numbers"
    input_dict = {"input": input_tensor, "pattern": pattern, "rewrite": rewrite, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array(["  leading and trailing spaces  "]).astype(np.string_)
    pattern = np.array(r"^\s+|\s+$").astype(np.string_)
    rewrite = np.array("").astype(np.string_)
    replace_global = True
    name = None
    input_dict = {"input": input_tensor, "pattern": pattern, "rewrite": rewrite, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array("MixedCaseString").astype(np.string_)
    pattern = np.array(r"([A-Z])").astype(np.string_)
    rewrite = np.array(r"_\1").astype(np.string_)
    replace_global = True
    name = "insert_underscore"
    input_dict = {"input": input_tensor, "pattern": pattern, "rewrite": rewrite, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 Removed as it causes error

    # Input 8
    input_tensor = np.array(["apple,banana,orange"]).astype(np.string_)
    pattern = np.array(",").astype(np.string_)
    rewrite = np.array("; ").astype(np.string_)
    replace_global = True
    name = "replace_comma"
    input_dict = {"input": input_tensor, "pattern": pattern, "rewrite": rewrite, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array(["abc123def", "ghi456jkl"]).astype(np.string_)
    pattern = np.array(r"[a-z]").astype(np.string_)
    rewrite = np.array("*").astype(np.string_)
    replace_global = True
    name = None
    input_dict = {"input": input_tensor, "pattern": pattern, "rewrite": rewrite, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array("This is a test string.").astype(np.string_)
    pattern = np.array(r"\s+").astype(np.string_)
    rewrite = np.array(" ").astype(np.string_)
    replace_global = True
    name = "remove_extra_spaces"
    input_dict = {"input": input_tensor, "pattern": pattern, "rewrite": rewrite, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_tensor = np.array([["test1 test2", "test3 test4"], ["test5 test6", "test7 test8"]]).astype(np.string_)
    pattern = np.array(r"test(\d)").astype(np.string_)
    rewrite = np.array(r"new\1").astype(np.string_)
    replace_global = True
    name = None
    input_dict = {"input": input_tensor, "pattern": pattern, "rewrite": rewrite, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_tensor = np.array("hello world").astype(np.string_)
    pattern = np.array(r"(hello) (world)").astype(np.string_)
    rewrite = np.array(r"\2 \1").astype(np.string_)
    replace_global = True
    name = "swap_words"
    input_dict = {"input": input_tensor, "pattern": pattern, "rewrite": rewrite, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.regex_replace"] = tf_strings_regex_replace_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.regex_replace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.regex_replace'.")

check_valid('tf.strings.regex_replace', generated_inputs['tf.strings.regex_replace'], lib="tf", suffix=0)
