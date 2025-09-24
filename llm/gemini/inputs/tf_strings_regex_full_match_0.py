
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_regex_full_match_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array(["hello world", "regex match", "world hello"], dtype=np.object_)
    pattern = np.array("hello.*", dtype=np.object_)
    name = None
    input_dict = {"input": input_tensor, "pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array(["12345", "67890", "abcde"], dtype=np.object_)
    pattern = np.array("^[0-9]+$", dtype=np.object_)
    name = "numeric_check"
    input_dict = {"input": input_tensor, "pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array(["apple", "banana", "orange"], dtype=np.object_)
    pattern = np.array("apple|banana", dtype=np.object_)
    name = None
    input_dict = {"input": input_tensor, "pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array(["a", "aa", "aaa"], dtype=np.object_)
    pattern = np.array("a+", dtype=np.object_)
    name = None
    input_dict = {"input": input_tensor, "pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array(["", " "], dtype=np.object_)
    pattern = np.array("^\\s*$", dtype=np.object_)
    name = None
    input_dict = {"input": input_tensor, "pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array(["test1", "test2", "test3"], dtype=np.object_)
    pattern = np.array("test[0-9]", dtype=np.object_)
    name = None
    input_dict = {"input": input_tensor, "pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array(["abc", "def", "ghi"], dtype=np.object_)
    pattern = np.array("^(abc|def|ghi)$", dtype=np.object_)
    name = None
    input_dict = {"input": input_tensor, "pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array(["a.b", "a_b", "a-b"], dtype=np.object_)
    pattern = np.array("a[._-]b", dtype=np.object_)
    name = None
    input_dict = {"input": input_tensor, "pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array(["file.txt", "image.png", "data.csv"], dtype=np.object_)
    pattern = np.array(".*\\.txt$", dtype=np.object_)
    name = None
    input_dict = {"input": input_tensor, "pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array(["123-456-7890", "1234567890", "(123) 456-7890"], dtype=np.object_)
    pattern = np.array("^\\d{3}[- ]?\\d{3}[- ]?\\d{4}$", dtype=np.object_)
    name = None
    input_dict = {"input": input_tensor, "pattern": pattern, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.regex_full_match"] = tf_strings_regex_full_match_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.regex_full_match' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.regex_full_match'.")

check_valid('tf.strings.regex_full_match', generated_inputs['tf.strings.regex_full_match'], lib="tf", suffix=0)
