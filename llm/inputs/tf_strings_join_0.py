
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_join_inputs():
    list_of_inputs = []

    # Input 1
    inputs = [np.array(['abc', 'def'])]
    separator = ''
    name = None
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = [np.array([['abc', '123'], ['def', '456'], ['ghi', '789']])]
    separator = ''
    name = None
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = [np.array([['abc', '123'], ['def', '456']])]
    separator = ' '
    name = None
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = [np.array(['hello', 'world']), np.array(['!', '!'])]
    separator = ''
    name = None
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = [np.array(['hello', 'world']), np.array(['!', '!'])]
    separator = ' '
    name = 'join_example'
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = [np.array([['a', 'b'], ['c', 'd']]), np.array([['1', '2'], ['3', '4']])]
    separator = '-'
    name = None
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = [np.array([[['a', 'b'], ['c', 'd']]]), np.array([[['1', '2'], ['3', '4']]])]
    separator = '+'
    name = '3d_join'
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    inputs = [np.array(['', ''])]
    separator = 'sep'
    name = None
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = [np.array([['', ''], ['', '']])]
    separator = 'sep'
    name = None
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    inputs = [np.array(['abc', 'def', 'ghi']), np.array(['123', '456', '789']), np.array(['jkl', 'mno', 'pqr'])]
    separator = ','
    name = 'many_inputs'
    input_dict = {'inputs': inputs, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}

def process_inputs():
    global generated_inputs
    input_list = tf_strings_join_inputs()
    
    processed_list = []
    for input_dict in input_list:
        processed_input = {}
        processed_input['inputs'] = [tf.convert_to_tensor(x, dtype=tf.string) for x in input_dict['inputs']]
        processed_input['separator'] = input_dict['separator']
        processed_input['name'] = input_dict['name']
        processed_list.append(processed_input)
        
    generated_inputs["tf.strings.join"] = processed_list

process_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.join' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.join'.")

check_valid('tf.strings.join', generated_inputs['tf.strings.join'], lib="tf", suffix=0)
