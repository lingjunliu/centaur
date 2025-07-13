
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_unicode_split_with_offsets_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([s.encode('utf8') for s in (u'G\xf6\xf6dnight', u'\U0001f60a')], dtype=np.object_)
    input_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    name = "unicode_split_1"
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([s.encode('utf16be') for s in (u'Hello', u'World')], dtype=np.object_)
    input_encoding = "UTF-16-BE"
    errors = "strict"
    replacement_char = 0
    name = "unicode_split_2"
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([s.encode('ascii', 'ignore') for s in (u'Ascii123', u'Another')], dtype=np.object_)
    input_encoding = "ASCII"
    errors = "ignore"
    replacement_char = 63
    name = "unicode_split_3"
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([u'你好世界'.encode('utf8')], dtype=np.object_)
    input_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([s.encode('utf8') for s in (u'test', u'with\nnewline')], dtype=np.object_)
    input_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    name = "unicode_split_5"
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([b''], dtype=np.object_)
    input_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    name = "unicode_split_6"
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([s.encode('utf32be') for s in (u'A', u'B')], dtype=np.object_)
    input_encoding = "UTF-32-BE"
    errors = "replace"
    replacement_char = 0
    name = "unicode_split_7"
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([s.encode('latin-1') for s in (u'latin', u'string')], dtype=np.object_)
    input_encoding = "latin-1"
    errors = "replace"
    replacement_char = 65533
    name = "unicode_split_8"
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: Multidimensional input
    input_tensor = np.array([[s.encode('utf8') for s in (u'hello', u'world')], [s.encode('utf8') for s in (u'foo', u'bar')]], dtype=np.object_)
    input_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    name = "unicode_split_9"
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different replacement char
    input_tensor = np.array([s.encode('utf8') for s in (u'G\xf6\xf6dnight', u'\U0001f60a')], dtype=np.object_)
    input_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 42  # Replace with asterisk
    name = "unicode_split_10"
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: errors='ignore'
    input_tensor = np.array([b'\xFF'], dtype=np.object_) # Invalid UTF-8 sequence
    input_encoding = "UTF-8"
    errors = "ignore"
    replacement_char = 65533
    name = "unicode_split_11"
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.unicode_split_with_offsets"] = tf_strings_unicode_split_with_offsets_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.unicode_split_with_offsets' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.unicode_split_with_offsets'.")

check_valid('tf.strings.unicode_split_with_offsets', generated_inputs['tf.strings.unicode_split_with_offsets'], lib="tf", suffix=0)
