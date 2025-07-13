
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_regexfullmatch_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array(["hello world", "regex match"], dtype=np.string_)
    pattern_tensor = np.array("^hello.*$", dtype=np.string_)
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array(["12345", "abcde"], dtype=np.string_)
    pattern_tensor = np.array("^[0-9]+$", dtype=np.string_)
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "name": "numeric_check"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array(["test@example.com", "invalid-email"], dtype=np.string_)
    pattern_tensor = np.array("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$", dtype=np.string_)
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (empty string)
    input_tensor = np.array(["", "non-empty"], dtype=np.string_)
    pattern_tensor = np.array("^$", dtype=np.string_)
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (multiline)
    input_tensor = np.array(["line1\nline2", "single line"], dtype=np.string_)
    pattern_tensor = np.array("line1.*line2", dtype=np.string_)
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (2D array)
    input_tensor = np.array([["abc", "def"], ["ghi", "jkl"]], dtype=np.string_)
    pattern_tensor = np.array("^[a-z]+$", dtype=np.string_)
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array(["a+b", "a*b"], dtype=np.string_)
    pattern_tensor = np.array("^a\\+b$", dtype=np.string_)
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (empty input)
    input_tensor = np.array([], dtype=np.string_)
    pattern_tensor = np.array("^.*$", dtype=np.string_)
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array(["  leading space", "trailing space  "], dtype=np.string_)
    pattern_tensor = np.array("^\\s*\\S.*\\S\\s*$", dtype=np.string_)
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (case insensitive)
    input_tensor = np.array(["Hello", "hello"], dtype=np.string_)
    pattern_tensor = np.array("(?i)^hello$", dtype=np.string_)
    input_dict = {"input": input_tensor, "pattern": pattern_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RegexFullMatch"] = tf_raw_ops_regexfullmatch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RegexFullMatch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RegexFullMatch'.")

check_valid('tf.raw_ops.RegexFullMatch', generated_inputs['tf.raw_ops.RegexFullMatch'], lib="tf", suffix=0)
