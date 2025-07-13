
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_unicode_decode_with_offsets_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([s.encode('utf8') for s in (u'G\xf6\xf6dnight', u'\U0001f60a')], dtype=np.object_)
    input_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": tf.constant(input_tensor),
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([s.encode('utf16') for s in (u'Hello', u'World')], dtype=np.object_)
    input_encoding = "UTF-16"
    errors = "strict"
    replacement_char = 0
    replace_control_characters = True
    name = "decode_op"

    input_dict = {
        "input": tf.constant(input_tensor),
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array(["abc".encode('ascii'), "def".encode('ascii')], dtype=np.object_)
    input_encoding = "ASCII"
    errors = "ignore"
    replacement_char = 63
    replace_control_characters = False
    name = None

    input_dict = {
        "input": tf.constant(input_tensor),
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([s.encode('latin1') for s in (u'café', u'thé')], dtype=np.object_)
    input_encoding = "Latin-1"
    errors = "replace"
    replacement_char = 42
    replace_control_characters = True
    name = None

    input_dict = {
        "input": tf.constant(input_tensor),
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array(["".encode('utf8'), "a".encode('utf8'), "b".encode('utf8')], dtype=np.object_)
    input_encoding = "UTF-8"
    errors = "strict"
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": tf.constant(input_tensor),
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (empty tensor)
    input_tensor = np.array([], dtype=np.object_)
    input_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.string),
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([b'\x01\x02', b'\x03\x04'], dtype=np.object_)
    input_encoding = 'raw_unicode_escape'
    errors = 'replace'
    replacement_char = 65533
    replace_control_characters = True
    name = None
    input_dict = {
        "input": tf.constant(input_tensor),
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8 (multidimensional tensor)
    input_tensor = np.array([[b'hello', b'world'], [b'foo', b'bar']], dtype=np.object_)
    input_encoding = 'utf-8'
    errors = 'ignore'
    replacement_char = 65533
    replace_control_characters = False
    name = None
    input_dict = {
        "input": tf.constant(input_tensor),
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (invalid utf-8 sequence)
    input_tensor = np.array([b'\xff'], dtype=np.object_)
    input_encoding = 'utf-8'
    errors = 'replace'
    replacement_char = 65533
    replace_control_characters = False
    name = None
    input_dict = {
        "input": tf.constant(input_tensor),
        "input_encoding": input_encoding,
        "errors": errors,
        "replacement_char": replacement_char,
        "replace_control_characters": replace_control_characters,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 with unicode characters
    input_tensor = np.array([u"你好世界".encode("utf-8"), u"こんにちは世界".encode("utf-8")], dtype=np.object_)
    input_encoding = 'utf-8'
    errors = 'replace'
    replacement_char = 65533
    replace_control_characters = False
    name = None

    input_dict = {
        "input": tf.constant(input_tensor),
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
