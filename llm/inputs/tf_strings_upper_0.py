
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_upper_inputs():
    list_of_inputs = []

    # Input 1: Basic string
    input_tensor = np.array("hello world")
    encoding = ""
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: String with numbers and special characters
    input_tensor = np.array("123 abc!@#")
    encoding = ""
    name = "string_with_numbers"
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty string
    input_tensor = np.array("")
    encoding = ""
    name = "empty_string"
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: String with UTF-8 encoding
    input_tensor = np.array("你好世界")
    encoding = "utf-8"
    name = "utf8_string"
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: String with mixed case
    input_tensor = np.array("MiXeD CaSe StRiNg")
    encoding = ""
    name = "mixed_case"
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: String with leading/trailing spaces
    input_tensor = np.array("  leading and trailing spaces  ")
    encoding = ""
    name = "spaces"
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multidimensional tensor (1D)
    input_tensor = np.array(["hello", "world", "tensorflow"])
    encoding = ""
    name = "1d_tensor"
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multidimensional tensor (2D)
    input_tensor = np.array([["hello", "world"], ["tensorflow", "rocks"]])
    encoding = ""
    name = "2d_tensor"
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: String with a very long sequence of characters.
    long_string = "a" * 200
    input_tensor = np.array(long_string)
    encoding = ""
    name = "long_string"
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array(["a", "b"])
    encoding = ""
    name = "short_string"
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.upper"] = tf_strings_upper_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.upper' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.upper'.")

check_valid('tf.strings.upper', generated_inputs['tf.strings.upper'], lib="tf", suffix=0)
