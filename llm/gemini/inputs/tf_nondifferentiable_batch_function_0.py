
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nondifferentiable_batch_function_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 4,
        'batch_timeout_micros': 1000,
        'allowed_batch_sizes': [1, 2, 3, 4],
        'max_enqueued_batches': 2,
        'autograph': True,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'num_batch_threads': 2,
        'max_batch_size': 8,
        'batch_timeout_micros': 5000,
        'allowed_batch_sizes': [1, 2, 4, 8],
        'max_enqueued_batches': 5,
        'autograph': False,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 16,
        'batch_timeout_micros': 10000,
        'allowed_batch_sizes': [1, 2, 4, 8, 16],
        'max_enqueued_batches': 3,
        'autograph': True,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'num_batch_threads': 2,
        'max_batch_size': 32,
        'batch_timeout_micros': 2000,
        'allowed_batch_sizes': [1, 2, 4, 8, 16, 32],
        'max_enqueued_batches': 1,
        'autograph': False,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 64,
        'batch_timeout_micros': 7000,
        'allowed_batch_sizes': [1, 2, 4, 8, 16, 32, 64],
        'max_enqueued_batches': 2,
        'autograph': True,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'num_batch_threads': 2,
        'max_batch_size': 128,
        'batch_timeout_micros': 15000,
        'allowed_batch_sizes': [1, 2, 4, 8, 16, 32, 64, 128],
        'max_enqueued_batches': 4,
        'autograph': False,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 5,
        'batch_timeout_micros': 500,
        'allowed_batch_sizes': [1, 2, 3, 4, 5],
        'max_enqueued_batches': 1,
        'autograph': True,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'num_batch_threads': 2,
        'max_batch_size': 10,
        'batch_timeout_micros': 2500,
        'allowed_batch_sizes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'max_enqueued_batches': 2,
        'autograph': False,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 20,
        'batch_timeout_micros': 12000,
        'allowed_batch_sizes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'max_enqueued_batches': 1,
        'autograph': True,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'num_batch_threads': 2,
        'max_batch_size': 256,
        'batch_timeout_micros': 100000,
        'allowed_batch_sizes': [1, 2, 4, 8, 16, 32, 64, 128, 256],
        'max_enqueued_batches': 2,
        'autograph': False,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nondifferentiable_batch_function"] = tf_nondifferentiable_batch_function_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nondifferentiable_batch_function' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nondifferentiable_batch_function'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nondifferentiable_batch_function', generated_inputs['tf.nondifferentiable_batch_function'], lib="tf", suffix=0)
