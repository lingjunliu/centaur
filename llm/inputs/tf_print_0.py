
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import sys
import copy
import logging
import absl.logging

def tf_print_inputs():
    list_of_inputs = []

    # Input 1: Basic test with sys.stdout
    input_dict = {
        'inputs': [np.array([1, 2, 3, 4, 5])],
        'output_stream': 'sys.stdout',
        'summarize': None,
        'sep': ' ',
        'end': '\n',
        'name': 'print_basic'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Test with sys.stderr and custom separator
    input_dict = {
        'inputs': [np.array([[1, 2], [3, 4]]), np.array([5, 6])],
        'output_stream': 'sys.stderr',
        'summarize': 2,
        'sep': ',',
        'end': '!',
        'name': 'print_stderr'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Test with a different summarization
    input_dict = {
        'inputs': [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])],
        'output_stream': 'sys.stdout',
        'summarize': -1,
        'sep': ' ',
        'end': '\n',
        'name': 'print_summarize_all'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-input test
    input_dict = {
        'inputs': [np.array([1, 2]), "hello", np.array([3, 4])],
        'output_stream': 'sys.stdout',
        'summarize': None,
        'sep': ' ',
        'end': '\n',
        'name': 'print_multi_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different summarize value
    input_dict = {
        'inputs': [np.array([[1, 2, 3, 4], [5, 6, 7, 8]])],
        'output_stream': 'sys.stdout',
        'summarize': 1,
        'sep': ' ',
        'end': '\n',
        'name': 'print_summarize_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: Empty numpy array
    input_dict = {
        'inputs': [np.array([])],
        'output_stream': 'sys.stdout',
        'summarize': None,
        'sep': ' ',
        'end': '\n',
        'name': 'print_empty_array'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Test with 3D array and Summarize -1
    input_dict = {
        'inputs': [np.arange(24).reshape((2, 3, 4))],
        'output_stream': 'sys.stdout',
        'summarize': -1,
        'sep': ' ',
        'end': '\n',
        'name': 'print_3d_array'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8:  tf.compat.v1.logging.error
    input_dict = {
        'inputs': [np.array([1, 2, 3])],
        'output_stream': 'tf.compat.v1.logging.error',
        'summarize': None,
        'sep': ' ',
        'end': '\n',
        'name': 'print_logging_error'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'inputs': [np.array([4, 5, 6])],
        'output_stream': 'absl.logging.warning',
        'summarize': None,
        'sep': ' ',
        'end': '\n',
        'name': 'print_absl_warning'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'inputs': [np.array([1, 2, 3])],
        'output_stream': 'tf.compat.v1.logging.info',
        'summarize': None,
        'sep': ' ',
        'end': '\n',
        'name': 'print_logging_info'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        'inputs': [np.array([7, 8, 9])],
        'output_stream': 'sys.stderr',
        'summarize': None,
        'sep': ' ',
        'end': '\n',
        'name': 'print_stderr_2'
    }
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
