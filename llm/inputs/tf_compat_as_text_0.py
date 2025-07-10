
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_compat_as_text_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "bytes_or_text": "hello world",
        "encoding": "utf-8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "bytes_or_text": "你好世界",
        "encoding": "utf-8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "bytes_or_text": "12345",
        "encoding": "utf-8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "bytes_or_text": "",
        "encoding": "utf-8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "bytes_or_text": "special characters: !@#$%^&*()_+=-`~[]{}|;':\",./<>?",
        "encoding": "utf-8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "bytes_or_text": "with newline\ncharacters",
        "encoding": "utf-8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "bytes_or_text": "with tabs\tcharacters",
        "encoding": "utf-8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "bytes_or_text": "mixed English and Chinese 你好世界",
        "encoding": "utf-8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "bytes_or_text": "some unicode characters: ☃★♲",
        "encoding": "utf-8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "bytes_or_text": "this is a longer string " * 20,
        "encoding": "utf-8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.compat.as_text"] = tf_compat_as_text_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.compat.as_text' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.compat.as_text'.")

check_valid('tf.compat.as_text', generated_inputs['tf.compat.as_text'], lib="tf", suffix=0)
