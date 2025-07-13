
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_regex_replace_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array("hello world", dtype=np.string_)
    pattern_tensor = np.array("world", dtype=np.string_)
    rewrite_tensor = np.array("tensorflow", dtype=np.string_)
    replace_global = True
    name = "replace_world"
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array("hello hello hello", dtype=np.string_)
    pattern_tensor = np.array("hello", dtype=np.string_)
    rewrite_tensor = np.array("hi", dtype=np.string_)
    replace_global = False
    name = "replace_first"
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array(["hello world", "foo bar"], dtype=np.string_)
    pattern_tensor = np.array("o", dtype=np.string_)
    rewrite_tensor = np.array("0", dtype=np.string_)
    replace_global = True
    name = "replace_o"
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array("a b c a b c", dtype=np.string_)
    pattern_tensor = np.array("a", dtype=np.string_)
    rewrite_tensor = np.array("d", dtype=np.string_)
    replace_global = True
    name = None
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array("123456789", dtype=np.string_)
    pattern_tensor = np.array("[0-9]", dtype=np.string_)
    rewrite_tensor = np.array("x", dtype=np.string_)
    replace_global = False
    name = "replace_digit"
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array("abc def ghi", dtype=np.string_)
    pattern_tensor = np.array("\\s+", dtype=np.string_)
    rewrite_tensor = np.array("_", dtype=np.string_)
    replace_global = True
    name = "replace_space"
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array(["test1", "test2", "test3"], dtype=np.string_)
    pattern_tensor = np.array("test", dtype=np.string_)
    rewrite_tensor = np.array("new_test", dtype=np.string_)
    replace_global = True
    name = "replace_test"
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array("multiple\nlines\nstring", dtype=np.string_)
    pattern_tensor = np.array("\n", dtype=np.string_)
    rewrite_tensor = np.array(" ", dtype=np.string_)
    replace_global = True
    name = "replace_newline"
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    input_tensor = np.array("Hello World Hello", dtype=np.string_)
    pattern_tensor = np.array("Hello", dtype=np.string_)
    rewrite_tensor = np.array("", dtype=np.string_)
    replace_global = True
    name = "remove_hello"
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array("1a2b3c4d", dtype=np.string_)
    pattern_tensor = np.array("[a-d]", dtype=np.string_)
    rewrite_tensor = np.array("*", dtype=np.string_)
    replace_global = True
    name = "replace_chars"
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "rewrite": rewrite_tensor, "replace_global": replace_global, "name": name}
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
