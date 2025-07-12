
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_nondifferentiable_batch_function_inputs():
    list_of_inputs = []

    def dummy_func(x):
      return x * 2

    # Input 1
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 4,
        'batch_timeout_micros': 1000,
        'allowed_batch_sizes': [2, 4],
        'max_enqueued_batches': 5,
        'autograph': True,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'num_batch_threads': 2,
        'max_batch_size': 8,
        'batch_timeout_micros': 500,
        'allowed_batch_sizes': [4, 8],
        'max_enqueued_batches': 10,
        'autograph': False,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'num_batch_threads': 4,
        'max_batch_size': 16,
        'batch_timeout_micros': 2000,
        'allowed_batch_sizes': [8, 16],
        'max_enqueued_batches': 3,
        'autograph': True,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 32,
        'batch_timeout_micros': 100,
        'allowed_batch_sizes': None,
        'max_enqueued_batches': 1,
        'autograph': False,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'num_batch_threads': 8,
        'max_batch_size': 64,
        'batch_timeout_micros': 4000,
        'allowed_batch_sizes': None,
        'max_enqueued_batches': 15,
        'autograph': True,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'num_batch_threads': 3,
        'max_batch_size': 128,
        'batch_timeout_micros': 750,
        'allowed_batch_sizes': None,
        'max_enqueued_batches': 7,
        'autograph': False,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'num_batch_threads': 5,
        'max_batch_size': 256,
        'batch_timeout_micros': 1500,
        'allowed_batch_sizes': None,
        'max_enqueued_batches': 9,
        'autograph': True,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'num_batch_threads': 10,
        'max_batch_size': 512,
        'batch_timeout_micros': 3000,
        'allowed_batch_sizes': None,
        'max_enqueued_batches': 2,
        'autograph': False,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 1024,
        'batch_timeout_micros': 5000,
        'allowed_batch_sizes': None,
        'max_enqueued_batches': 6,
        'autograph': True,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'num_batch_threads': 6,
        'max_batch_size': 2048,
        'batch_timeout_micros': 1250,
        'allowed_batch_sizes': None,
        'max_enqueued_batches': 12,
        'autograph': False,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nondifferentiable_batch_function"] = tf_nondifferentiable_batch_function_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nondifferentiable_batch_function' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nondifferentiable_batch_function'.")

check_valid('tf.nondifferentiable_batch_function', generated_inputs['tf.nondifferentiable_batch_function'], lib="tf", suffix=0)
