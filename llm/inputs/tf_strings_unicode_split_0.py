
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_unicode_split_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([s.encode('utf8') for s in (u'G\xf6\xf6dnight', u'\U0001f60a')], dtype=object)
    input_encoding = 'UTF-8'
    errors = 'replace'
    replacement_char = 65533
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([b'hello', b'world'], dtype=object)
    input_encoding = 'UTF-8'
    errors = 'strict'
    replacement_char = 65533
    name = 'test_split'
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([b''], dtype=object)
    input_encoding = 'UTF-8'
    errors = 'ignore'
    replacement_char = 0
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([s.encode('utf16') for s in (u'G\xf6\xf6dnight', u'\U0001f60a')], dtype=object)
    input_encoding = 'UTF-16'
    errors = 'replace'
    replacement_char = 65533
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([b'\xf0\x9f\x98'], dtype=object)
    input_encoding = 'UTF-8'
    errors = 'ignore'
    replacement_char = 65533
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    input_tensor = np.array([s.encode('latin-1') for s in (u'H\xe9llo', u'World')], dtype=object)
    input_encoding = 'UTF-8'
    errors = 'replace'
    replacement_char = 65533
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (multidimensional)
    input_tensor = np.array([[b'hello', b'world'], [b'foo', b'bar']], dtype=object)
    input_encoding = 'UTF-8'
    errors = 'strict'
    replacement_char = 65533
    name = 'test_split'
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([s.encode('utf8') for s in (u'你好世界', u'こんにちは世界')], dtype=object)
    input_encoding = 'UTF-8'
    errors = 'replace'
    replacement_char = 65533
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([b'\xf4\x80\x80\x80'], dtype=object) # Invalid UTF-8
    input_encoding = 'UTF-8'
    errors = 'replace'
    replacement_char = 65533
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([s.encode('utf32') for s in (u'A', u'B')], dtype=object)
    input_encoding = 'UTF-32'
    errors = 'replace'
    replacement_char = 65533
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.unicode_split"] = tf_strings_unicode_split_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.unicode_split' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.unicode_split'.")

check_valid('tf.strings.unicode_split', generated_inputs['tf.strings.unicode_split'], lib="tf", suffix=0)
