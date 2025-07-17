
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_regex_replace_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = "Text with tags.<br /><b>contains html</b>"
    pattern = "<[^>]+>"
    rewrite = " "
    replace_global = True
    name = None

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.string),
        "pattern": pattern,
        "rewrite": rewrite,
        "replace_global": replace_global,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = ["apple", "banana", "orange"]
    pattern = "a"
    rewrite = "A"
    replace_global = False
    name = "replace_a"

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.string),
        "pattern": pattern,
        "rewrite": rewrite,
        "replace_global": replace_global,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = [["hello world", "goodbye world"], ["foo bar", "baz qux"]]
    pattern = "world"
    rewrite = "universe"
    replace_global = True
    name = None

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.string),
        "pattern": pattern,
        "rewrite": rewrite,
        "replace_global": replace_global,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = "123-456-7890"
    pattern = r"(\d{3})-(\d{3})-(\d{4})"
    rewrite = r"(\1)\2-\3"
    replace_global = True
    name = None

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.string),
        "pattern": pattern,
        "rewrite": rewrite,
        "replace_global": replace_global,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = ["abc123def", "ghi456jkl", "mno789pqr"]
    pattern = r"\d+"
    rewrite = "NUM"
    replace_global = True
    name = "replace_digits"

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.string),
        "pattern": pattern,
        "rewrite": rewrite,
        "replace_global": replace_global,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    input_tensor = "This is a test string with multiple spaces."
    pattern = r"\s+"
    rewrite = " "
    replace_global = True
    name = None

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.string),
        "pattern": pattern,
        "rewrite": rewrite,
        "replace_global": replace_global,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = ["a", "aa", "aaa", "aaaa"]
    pattern = "a"
    rewrite = "b"
    replace_global = True
    name = None

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.string),
        "pattern": pattern,
        "rewrite": rewrite,
        "replace_global": replace_global,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = [["1", "2"], ["3", "4"]]
    pattern = r"\d"
    rewrite = "x"
    replace_global = True
    name = "replace_number"
    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.string),
        "pattern": pattern,
        "rewrite": rewrite,
        "replace_global": replace_global,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = "Hello, world!"
    pattern = r"[aeiouAEIOU]"
    rewrite = "*"
    replace_global = True
    name = None
    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.string),
        "pattern": pattern,
        "rewrite": rewrite,
        "replace_global": replace_global,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = ["test1", "test2", "test3"]
    pattern = r"test(\d)"
    rewrite = r"result\1"
    replace_global = True
    name = None
    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.string),
        "pattern": pattern,
        "rewrite": rewrite,
        "replace_global": replace_global,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 - Testing replace_global = False
    input_tensor = "ababab"
    pattern = "a"
    rewrite = "X"
    replace_global = False
    name = None
    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.string),
        "pattern": pattern,
        "rewrite": rewrite,
        "replace_global": replace_global,
        "name": name
    }
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
