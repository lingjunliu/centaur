
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_unicode_split_with_offsets_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = tf.constant(["hello", "world"]).numpy()
    input_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([s.encode('utf8') for s in (u'G\xf6\xf6dnight', u'\U0001f60a')], dtype=np.string_)
    input_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    name = "split_name"
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = tf.constant(["你好世界", "你好"]).numpy()
    input_encoding = "UTF-8"
    errors = "ignore"
    replacement_char = 65533
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = tf.constant(["abc\x80def"]).numpy() # Invalid UTF-8
    input_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = tf.constant(["abc\x80def"]).numpy()
    input_encoding = "UTF-8"
    errors = "ignore"
    replacement_char = 65533
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = tf.constant(["hello", "你好"]).numpy()
    input_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 0 # Different replacement char
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Scalar input
    input_tensor = np.array(["你好".encode('utf-8')], dtype=np.string_)
    input_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8 - Empty string
    input_tensor = tf.constant([""]).numpy()
    input_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - Multi-dimensional tensor
    input_tensor = tf.constant([["hello", "world"], ["你好", "世界"]]).numpy()
    input_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - latin-1 encoding
    input_tensor = np.array([b'caf\xe9'], dtype=np.string_)
    input_encoding = "latin-1"
    errors = "replace"
    replacement_char = 65533
    name = None
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
