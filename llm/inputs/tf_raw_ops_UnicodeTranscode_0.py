
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_unicode_transcode_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = tf.constant(["Hello", "TensorFlow", "2.x"])
    input_encoding_str = "UTF-8"
    output_encoding_str = "UTF-16-BE"
    errors_str = "replace"
    replacement_char_int = 65533
    replace_control_characters_bool = False
    name_str = None

    input_dict = {
        "input": input_tensor,
        "input_encoding": input_encoding_str,
        "output_encoding": output_encoding_str,
        "errors": errors_str,
        "replacement_char": replacement_char_int,
        "replace_control_characters": replace_control_characters_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = tf.constant(["A", "B", "C"])
    input_encoding_str = "US ASCII"
    output_encoding_str = "UTF-8"
    errors_str = "strict"
    replacement_char_int = 65533
    replace_control_characters_bool = True
    name_str = "transcode_op"

    input_dict = {
        "input": input_tensor,
        "input_encoding": input_encoding_str,
        "output_encoding": output_encoding_str,
        "errors": errors_str,
        "replacement_char": replacement_char_int,
        "replace_control_characters": replace_control_characters_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = tf.constant(["你好", "世界"])
    input_encoding_str = "UTF-8"
    output_encoding_str = "UTF-8"
    errors_str = "ignore"
    replacement_char_int = 12345
    replace_control_characters_bool = False
    name_str = None

    input_dict = {
        "input": input_tensor,
        "input_encoding": input_encoding_str,
        "output_encoding": output_encoding_str,
        "errors": errors_str,
        "replacement_char": replacement_char_int,
        "replace_control_characters": replace_control_characters_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.UnicodeTranscode"] = tf_raw_ops_unicode_transcode_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.UnicodeTranscode' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.UnicodeTranscode'.")

check_valid('tf.raw_ops.UnicodeTranscode', generated_inputs['tf.raw_ops.UnicodeTranscode'], lib="tf", suffix=0)
