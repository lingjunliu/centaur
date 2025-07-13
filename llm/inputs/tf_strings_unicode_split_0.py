
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_unicode_split_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([s.encode('utf8') for s in (u'G\xf6\xf6dnight', u'\U0001f60a')])
    input_encoding = 'UTF-8'
    errors = 'replace'
    replacement_char = 65533
    name = None
    input_dict = {'input': tf.constant(input_tensor, dtype=tf.string), 'input_encoding': input_encoding, 'errors': errors, 'replacement_char': int(replacement_char), 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([s.encode('utf16') for s in (u'Hello', u'World')])
    input_encoding = 'UTF-16'
    errors = 'strict'
    replacement_char = 0
    name = 'test_name'
    input_dict = {'input': tf.constant(input_tensor, dtype=tf.string), 'input_encoding': input_encoding, 'errors': errors, 'replacement_char': int(replacement_char), 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([u'你好世界'.encode('utf8'), u'你好'.encode('utf8')])
    input_encoding = 'UTF-8'
    errors = 'ignore'
    replacement_char = 42
    name = None
    input_dict = {'input': tf.constant(input_tensor, dtype=tf.string), 'input_encoding': input_encoding, 'errors': errors, 'replacement_char': int(replacement_char), 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 - scalar input
    input_tensor = np.array(u'你好'.encode('utf8'))
    input_encoding = 'UTF-8'
    errors = 'replace'
    replacement_char = 65533
    name = None
    input_dict = {'input': tf.constant(input_tensor, dtype=tf.string), 'input_encoding': input_encoding, 'errors': errors, 'replacement_char': int(replacement_char), 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 - different replacement char
    input_tensor = np.array([s.encode('utf8') for s in (u'G\xf6\xf6dnight', u'\U0001f60a')])
    input_encoding = 'UTF-8'
    errors = 'replace'
    replacement_char = 1234
    name = None
    input_dict = {'input': tf.constant(input_tensor, dtype=tf.string), 'input_encoding': input_encoding, 'errors': errors, 'replacement_char': int(replacement_char), 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - empty strings
    input_tensor = np.array([''.encode('utf8'), ''.encode('utf8')])
    input_encoding = 'UTF-8'
    errors = 'replace'
    replacement_char = 65533
    name = None
    input_dict = {'input': tf.constant(input_tensor, dtype=tf.string), 'input_encoding': input_encoding, 'errors': errors, 'replacement_char': int(replacement_char), 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - multi-dimensional array
    input_tensor = np.array([[s.encode('utf8') for s in (u'G\xf6\xf6dnight', u'\U0001f60a')], [s.encode('utf8') for s in (u'Hello', u'World')]])
    input_encoding = 'UTF-8'
    errors = 'replace'
    replacement_char = 65533
    name = None
    input_dict = {'input': tf.constant(input_tensor, dtype=tf.string), 'input_encoding': input_encoding, 'errors': errors, 'replacement_char': int(replacement_char), 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - ASCII strings
    input_tensor = np.array(['Hello'.encode('ascii'), 'World'.encode('ascii')])
    input_encoding = 'ASCII'
    errors = 'replace'
    replacement_char = 65533
    name = None
    input_dict = {'input': tf.constant(input_tensor, dtype=tf.string), 'input_encoding': input_encoding, 'errors': errors, 'replacement_char': int(replacement_char), 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - latin-1 encoding
    input_tensor = np.array([u'café'.encode('latin-1'), u'España'.encode('latin-1')])
    input_encoding = 'latin-1'
    errors = 'replace'
    replacement_char = 65533
    name = None
    input_dict = {'input': tf.constant(input_tensor, dtype=tf.string), 'input_encoding': input_encoding, 'errors': errors, 'replacement_char': int(replacement_char), 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - Errors strict with invalid utf8
    input_tensor = np.array([b'\xff'])
    input_encoding = 'UTF-8'
    errors = 'replace'
    replacement_char = 65533
    name = None
    input_dict = {'input': tf.constant(input_tensor, dtype=tf.string), 'input_encoding': input_encoding, 'errors': errors, 'replacement_char': int(replacement_char), 'name': name}
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
