
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_UnicodeTranscode_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array(["Hello", "TensorFlow"], dtype=np.string_)
    input_encoding = "UTF-8"
    output_encoding = "UTF-16-BE"
    errors = "replace"
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": tf.constant(input_tensor),
        "input_encoding": input_encoding,
        "output_encoding": output_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array(["你好", "世界"], dtype=np.unicode_)
    input_encoding = "UTF-8"
    output_encoding = "UTF-8"
    errors = "strict"
    replacement_char = 65533
    replace_control_characters = True
    name = "transcode_2"
    input_tensor = tf.constant(input_tensor, dtype=tf.string)


    input_dict = {
        "input": input_tensor,
        "input_encoding": input_encoding,
        "output_encoding": output_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array(["A", "B", "C"], dtype=np.string_)
    input_encoding = "US ASCII"
    output_encoding = "UTF-8"
    errors = "ignore"
    replacement_char = 42
    replace_control_characters = False
    name = None

    input_dict = {
        "input": tf.constant(input_tensor),
        "input_encoding": input_encoding,
        "output_encoding": output_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array(["\x01\x02\x03", "\x04\x05\x06"], dtype=np.string_)
    input_encoding = "UTF-8"
    output_encoding = "UTF-16-BE"
    errors = "replace"
    replacement_char = 65533
    replace_control_characters = True
    name = None

    input_dict = {
        "input": tf.constant(input_tensor),
        "input_encoding": input_encoding,
        "output_encoding": output_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array(["Invalid UTF-8: \xf0\x9f\x98\x80".encode('utf-8')], dtype=np.string_)
    input_encoding = "UTF-8"
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": tf.constant(input_tensor),
        "input_encoding": input_encoding,
        "output_encoding": output_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    input_tensor = np.array(["test"], dtype=np.string_)
    input_encoding = "UTF-8"
    output_encoding = "UTF-32-BE"
    errors = "replace"
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": tf.constant(input_tensor),
        "input_encoding": input_encoding,
        "output_encoding": output_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty string
    input_tensor = np.array([""], dtype=np.string_)
    input_encoding = "UTF-8"
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": tf.constant(input_tensor),
        "input_encoding": input_encoding,
        "output_encoding": output_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multidimensional tensor
    input_tensor = np.array([["Hello", "World"], ["TensorFlow", "2.0"]], dtype=np.string_)
    input_encoding = "UTF-8"
    output_encoding = "UTF-16-BE"
    errors = "replace"
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": tf.constant(input_tensor),
        "input_encoding": input_encoding,
        "output_encoding": output_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array(["\x00\x41", "\x00\x42"], dtype=np.string_)
    input_encoding = "UTF-16-BE"
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": tf.constant(input_tensor),
        "input_encoding": input_encoding,
        "output_encoding": output_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Using small replacement char
    input_tensor = np.array(["Invalid UTF-8: \xf0\x9f\x98\x80".encode('utf-8')], dtype=np.string_)
    input_encoding = "UTF-8"
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 32
    replace_control_characters = False
    name = None

    input_dict = {
        "input": tf.constant(input_tensor),
        "input_encoding": input_encoding,
        "output_encoding": output_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.UnicodeTranscode"] = tf_raw_ops_UnicodeTranscode_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.UnicodeTranscode' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.UnicodeTranscode'.")

check_valid('tf.raw_ops.UnicodeTranscode', generated_inputs['tf.raw_ops.UnicodeTranscode'], lib="tf", suffix=0)
