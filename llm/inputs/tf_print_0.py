
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import sys
import absl.logging

def tf_print_inputs():
    list_of_inputs = []

    # Input 1: Basic example with stdout
    inputs = [np.array([1, 2, 3])]
    input_dict = {'inputs': inputs, 'output_stream': 'sys.stdout', 'summarize': 3, 'sep': ' ', 'end': '\n', 'name': 'print_op_1'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Using stderr, different separator, no summarization
    inputs = [np.array([4, 5, 6]), np.array([7, 8, 9])]
    input_dict = {'inputs': inputs, 'output_stream': 'sys.stderr', 'summarize': None, 'sep': ',', 'end': '!', 'name': 'print_op_2'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Printing a large array with summarization
    inputs = [np.arange(20)]
    input_dict = {'inputs': inputs, 'output_stream': 'sys.stdout', 'summarize': 5, 'sep': ' ', 'end': '\n', 'name': 'print_op_3'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Printing a multi-dimensional array
    inputs = [np.array([[1, 2], [3, 4], [5,6]])]
    input_dict = {'inputs': inputs, 'output_stream': 'sys.stdout', 'summarize': 2, 'sep': ' ', 'end': '\n', 'name': 'print_op_4'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Printing to a file
    inputs = [np.array([10, 11, 12])]
    input_dict = {'inputs': inputs, 'output_stream': 'file:///tmp/test_print.txt', 'summarize': 3, 'sep': ' ', 'end': '\n', 'name': 'print_op_5'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Using a negative summarize value to print all elements
    inputs = [np.array([13, 14, 15])]
    input_dict = {'inputs': inputs, 'output_stream': 'sys.stdout', 'summarize': -1, 'sep': ' ', 'end': '\n', 'name': 'print_op_6'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Printing a complex array
    inputs = [np.array([1 + 1j, 2 + 2j, 3 + 3j])]
    input_dict = {'inputs': inputs, 'output_stream': 'sys.stdout', 'summarize': 3, 'sep': ' ', 'end': '\n', 'name': 'print_op_7'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Using absl logging
    inputs = [np.array([16, 17, 18])]
    input_dict = {'inputs': inputs, 'output_stream': 'absl.logging.info', 'summarize': 3, 'sep': ' ', 'end': '\n', 'name': 'print_op_8'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.print"] = tf_print_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.print' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.print'.")

check_valid('tf.print', generated_inputs['tf.print'], lib="tf", suffix=0)
