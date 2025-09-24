
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_compat_as_str_inputs():
    list_of_inputs = []

    # Input 1: Basic UTF-8 string
    input_dict = {"bytes_or_text": "hello", "encoding": "utf-8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: UTF-8 string with special characters
    input_dict = {"bytes_or_text": "你好世界", "encoding": "utf-8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty UTF-8 string
    input_dict = {"bytes_or_text": "", "encoding": "utf-8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: ASCII string
    input_dict = {"bytes_or_text": "ASCII", "encoding": "ascii"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: String containing numbers
    input_dict = {"bytes_or_text": "12345", "encoding": "utf-8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: String containing a mix of characters, numbers, and symbols
    input_dict = {"bytes_or_text": "a1b2c3!@#", "encoding": "utf-8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Latin-1 encoding
    input_dict = {"bytes_or_text": "ISO-8859-1", "encoding": "latin-1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: String containing Unicode characters
    input_dict = {"bytes_or_text": "€æøå", "encoding": "utf-8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: String containing newline and tab characters
    input_dict = {"bytes_or_text": "line1\nline2\tTab", "encoding": "utf-8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Long String
    long_string = "a" * 200
    input_dict = {"bytes_or_text": long_string, "encoding": "utf-8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.compat.as_str"] = tf_compat_as_str_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.compat.as_str' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.compat.as_str'.")

check_valid('tf.compat.as_str', generated_inputs['tf.compat.as_str'], lib="tf", suffix=0)
