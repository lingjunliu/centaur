
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_unicode_decode_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([s.encode('utf8') for s in (u'G\xf6\xf6dnight', u'\U0001f60a')], dtype=np.object_)
    input_encoding = 'UTF-8'
    errors = 'replace'
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": input_tensor,
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([b'hello', b'world'], dtype=np.object_)
    input_encoding = 'ASCII'
    errors = 'strict'
    replacement_char = 63
    replace_control_characters = True
    name = 'decode_ascii'

    input_dict = {
        "input": input_tensor,
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([b'\x80abc', b'def'], dtype=np.object_)
    input_encoding = 'latin-1'
    errors = 'ignore'
    replacement_char = 0
    replace_control_characters = False
    name = None

    input_dict = {
        "input": input_tensor,
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multidimensional array
    input_tensor = np.array([[b'hello', b'world'], [b'foo', b'bar']], dtype=np.object_)
    input_encoding = 'ASCII'
    errors = 'replace'
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": input_tensor,
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty string
    input_tensor = np.array([b''], dtype=np.object_)
    input_encoding = 'UTF-8'
    errors = 'replace'
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": input_tensor,
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different encoding
    input_tensor = np.array([u'你好世界'.encode('gbk')], dtype=np.object_)
    input_encoding = 'GBK'
    errors = 'replace'
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": input_tensor,
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: Errors='strict' with invalid input
    input_tensor = np.array([b'\xff'], dtype=np.object_)
    input_encoding = 'latin-1'
    errors = 'replace'
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": input_tensor,
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([b'\x01\x02abc', b'def'], dtype=np.object_)
    input_encoding = 'ASCII'
    errors = 'replace'
    replacement_char = 63
    replace_control_characters = True
    name = None

    input_dict = {
        "input": input_tensor,
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([b'test\x00string'], dtype=np.object_)
    input_encoding = 'UTF-8'
    errors = 'replace'
    replacement_char = 65533
    replace_control_characters = True
    name = None

    input_dict = {
        "input": input_tensor,
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Scalar input
    input_tensor = np.array(b'hello', dtype=np.object_)
    input_encoding = 'ASCII'
    errors = 'replace'
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": input_tensor,
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.unicode_decode"] = tf_strings_unicode_decode_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.unicode_decode' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.unicode_decode'.")

check_valid('tf.strings.unicode_decode', generated_inputs['tf.strings.unicode_decode'], lib="tf", suffix=0)
