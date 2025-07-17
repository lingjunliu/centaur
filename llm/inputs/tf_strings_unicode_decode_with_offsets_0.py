
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_unicode_decode_with_offsets_inputs():
    list_of_inputs = []

    # Input 1
    input_str = np.array([b'G\xf6\xf6dnight', b'\xf0\x9f\x98\x8a'])
    input_encoding = 'UTF-8'
    errors = 'replace'
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": tf.convert_to_tensor(input_str, dtype=tf.string),
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_str = np.array([b'hello', b'world'])
    input_encoding = 'ASCII'
    errors = 'strict'
    replacement_char = 63
    replace_control_characters = True
    name = 'my_decode'

    input_dict = {
        "input": tf.convert_to_tensor(input_str, dtype=tf.string),
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_str = np.array([b'\x80abc', b'def'])
    input_encoding = 'GB18030'
    errors = 'ignore'
    replacement_char = 0
    replace_control_characters = False
    name = None

    input_dict = {
        "input": tf.convert_to_tensor(input_str, dtype=tf.string),
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_str = np.array([b'\x01\x02abc', b'def'])
    input_encoding = 'UTF-8'
    errors = 'replace'
    replacement_char = 65533
    replace_control_characters = True
    name = None
    input_dict = {
        "input": tf.convert_to_tensor(input_str, dtype=tf.string),
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 - scalar input
    input_str = np.array(b'test')
    input_encoding = 'UTF-8'
    errors = 'replace'
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": tf.convert_to_tensor(input_str, dtype=tf.string),
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - empty string
    input_str = np.array([b'', b''])
    input_encoding = 'UTF-8'
    errors = 'replace'
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": tf.convert_to_tensor(input_str, dtype=tf.string),
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - multidimensional
    input_str = np.array([[b'abc', b'def'], [b'ghi', b'jkl']])
    input_encoding = 'ASCII'
    errors = 'replace'
    replacement_char = 65533
    replace_control_characters = False
    name = None
    input_dict = {
        "input": tf.convert_to_tensor(input_str, dtype=tf.string),
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_str = np.array([b'G\xf6\xf6dnight', b'\xf0\x9f\x98\x8a'])
    input_encoding = 'UTF-8'
    errors = 'strict'
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": tf.convert_to_tensor(input_str, dtype=tf.string),
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_str = np.array([b'hello\x01', b'world'])
    input_encoding = 'ASCII'
    errors = 'replace'
    replacement_char = 63
    replace_control_characters = True
    name = 'my_decode2'

    input_dict = {
        "input": tf.convert_to_tensor(input_str, dtype=tf.string),
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_str = np.array([b'\x80abc', b'def'])
    input_encoding = 'GB18030'
    errors = 'replace'
    replacement_char = 12345
    replace_control_characters = False
    name = None

    input_dict = {
        "input": tf.convert_to_tensor(input_str, dtype=tf.string),
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.unicode_decode_with_offsets"] = tf_strings_unicode_decode_with_offsets_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.unicode_decode_with_offsets' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.unicode_decode_with_offsets'.")

check_valid('tf.strings.unicode_decode_with_offsets', generated_inputs['tf.strings.unicode_decode_with_offsets'], lib="tf", suffix=0)
