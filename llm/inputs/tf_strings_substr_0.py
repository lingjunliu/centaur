
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_substr_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([b'Hello', b'World'], dtype=np.object_)
    pos_tensor = np.array(1, dtype=np.int32)
    len_tensor = np.array(3, dtype=np.int32)
    unit_str = 'BYTE'
    name_str = 'substr_example1'
    input_dict = {'input': input_tensor, 'pos': pos_tensor, 'len': len_tensor, 'unit': unit_str, 'name': name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[b'ten', b'eleven'], [b'twelve', b'thirteen']], dtype=np.object_)
    pos_tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    len_tensor = np.array([[2, 3], [4, 5]], dtype=np.int32)
    unit_str = 'BYTE'
    name_str = 'substr_example2'
    input_dict = {'input': input_tensor, 'pos': pos_tensor, 'len': len_tensor, 'unit': unit_str, 'name': name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([b'abcdefg'], dtype=np.object_)
    pos_tensor = np.array(2, dtype=np.int32)
    len_tensor = np.array(4, dtype=np.int32)
    unit_str = 'BYTE'
    name_str = 'substr_example3'
    input_dict = {'input': input_tensor, 'pos': pos_tensor, 'len': len_tensor, 'unit': unit_str, 'name': name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 - negative pos
    input_tensor = np.array([b'abcdefg'], dtype=np.object_)
    pos_tensor = np.array(-3, dtype=np.int32)
    len_tensor = np.array(2, dtype=np.int32)
    unit_str = 'BYTE'
    name_str = 'substr_example4'
    input_dict = {'input': input_tensor, 'pos': pos_tensor, 'len': len_tensor, 'unit': unit_str, 'name': name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 - negative len (should take as much as possible)
    input_tensor = np.array([b'abcdefg'], dtype=np.object_)
    pos_tensor = np.array(2, dtype=np.int32)
    len_tensor = np.array(-1, dtype=np.int32)
    unit_str = 'BYTE'
    name_str = 'substr_example5'
    input_dict = {'input': input_tensor, 'pos': pos_tensor, 'len': len_tensor, 'unit': unit_str, 'name': name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - broadcasting pos and len
    input_tensor = np.array([[b'ten', b'eleven'], [b'twelve', b'thirteen']], dtype=np.object_)
    pos_tensor = np.array([1, 2], dtype=np.int32)
    len_tensor = np.array([2, 3], dtype=np.int32)
    unit_str = 'BYTE'
    name_str = 'substr_example6'
    input_dict = {'input': input_tensor, 'pos': pos_tensor, 'len': len_tensor, 'unit': unit_str, 'name': name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - broadcasting input
    input_tensor = np.array(b'thirteen', dtype=np.object_)
    pos_tensor = np.array([1, 5], dtype=np.int32)
    len_tensor = np.array([3, 2], dtype=np.int32)
    unit_str = 'BYTE'
    name_str = 'substr_example7'
    input_dict = {'input': input_tensor, 'pos': pos_tensor, 'len': len_tensor, 'unit': unit_str, 'name': name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - UTF8_CHAR
    input_tensor = np.array([b'H\xc3\xa9llo', b'W\xc3\xb6rld'], dtype=np.object_)
    pos_tensor = np.array(1, dtype=np.int32)
    len_tensor = np.array(3, dtype=np.int32)
    unit_str = 'UTF8_CHAR'
    name_str = 'substr_example8'
    input_dict = {'input': input_tensor, 'pos': pos_tensor, 'len': len_tensor, 'unit': unit_str, 'name': name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - UTF8_CHAR with broadcasting
    input_tensor = np.array([[b't\xc3\xa9n', b'el\xc3\xa9ven'], [b'twelve', b'thirteen']], dtype=np.object_)
    pos_tensor = np.array([[1, 0], [2, 1]], dtype=np.int32)
    len_tensor = np.array([[1, 2], [2, 3]], dtype=np.int32)
    unit_str = 'UTF8_CHAR'
    name_str = 'substr_example9'
    input_dict = {'input': input_tensor, 'pos': pos_tensor, 'len': len_tensor, 'unit': unit_str, 'name': name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - Empty string input
    input_tensor = np.array([b''], dtype=np.object_)
    pos_tensor = np.array(0, dtype=np.int32)
    len_tensor = np.array(0, dtype=np.int32)
    unit_str = 'BYTE'
    name_str = 'substr_example10'
    input_dict = {'input': input_tensor, 'pos': pos_tensor, 'len': len_tensor, 'unit': unit_str, 'name': name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.substr"] = tf_strings_substr_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.substr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.substr'.")

check_valid('tf.strings.substr', generated_inputs['tf.strings.substr'], lib="tf", suffix=0)
