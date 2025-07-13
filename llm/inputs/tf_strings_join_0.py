
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_join_inputs():
    list_of_inputs = []

    # Input 1
    inputs = [np.array(['abc', 'def'], dtype=np.str_)]
    separator = ''
    name = None
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = [np.array([['abc', '123'], ['def', '456']], dtype=np.str_)]
    separator = ' '
    name = 'join_op'
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = [np.array([['abc', '123']], dtype=np.str_), np.array([['def', '456']], dtype=np.str_)]
    separator = '-'
    name = None
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = [np.array(['hello', 'world'], dtype=np.str_), np.array(['!', '.'], dtype=np.str_)]
    separator = ' '
    name = 'my_join'
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = [np.array([[['a', 'b'], ['c', 'd']]], dtype=np.str_), np.array([[[ '1', '2'], ['3', '4']]], dtype=np.str_)]
    separator = ','
    name = None
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = [np.array([''], dtype=np.str_)]
    separator = 'sep'
    name = 'empty_string'
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    inputs = [np.array([['longstring1', 'longstring2']], dtype=np.str_)]
    separator = 'verylongseparator'
    name = 'long_strings'
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    inputs = [np.array([['','']], dtype=np.str_)]
    separator = ''
    name = None
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = [np.array(['string1', 'string2'], dtype=np.str_), np.array(['string3', 'string4'], dtype=np.str_), np.array(['string5', 'string6'], dtype=np.str_)]
    separator = '---'
    name = 'multiple_inputs'
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    inputs = [np.array([['s1', 's2'], ['s3', 's4']], dtype=np.str_), np.array([['s5', 's6'], ['s7', 's8']], dtype=np.str_)]
    separator = ''
    name = None
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
temp = tf_strings_join_inputs()
for i in range(len(temp)):
    temp_inputs = []
    for x in temp[i]['inputs']:
        temp_inputs.append(tf.convert_to_tensor(x))
    temp[i]['inputs'] = temp_inputs

generated_inputs["tf.strings.join"] = temp

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.join' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.join'.")

check_valid('tf.strings.join', generated_inputs['tf.strings.join'], lib="tf", suffix=0)
