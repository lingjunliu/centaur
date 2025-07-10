
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_nondifferentiable_batch_function_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 2,
        'batch_timeout_micros': 3,
        'allowed_batch_sizes': [2],
        'max_enqueued_batches': 10,
        'autograph': True,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'num_batch_threads': 2,
        'max_batch_size': 4,
        'batch_timeout_micros': 10,
        'allowed_batch_sizes': [2, 4],
        'max_enqueued_batches': 5,
        'autograph': False,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 5,
        'batch_timeout_micros': 100,
        'allowed_batch_sizes': [1,2,5],
        'max_enqueued_batches': 15,
        'autograph': True,
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
