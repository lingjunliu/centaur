
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_nondifferentiable_batch_function_inputs():
    list_of_inputs = []

    def dummy_func(x):
        return tf.matmul(x, x)

    # Input 1
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 2,
        'batch_timeout_micros': 3,
        'allowed_batch_sizes': [],
        'max_enqueued_batches': 10,
        'autograph': True,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'num_batch_threads': 2,
        'max_batch_size': 4,
        'batch_timeout_micros': 100,
        'allowed_batch_sizes': [2, 4],
        'max_enqueued_batches': 5,
        'autograph': False,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'num_batch_threads': 4,
        'max_batch_size': 8,
        'batch_timeout_micros': 500,
        'allowed_batch_sizes': [4, 8],
        'max_enqueued_batches': 15,
        'autograph': True,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 16,
        'batch_timeout_micros': 1000,
        'allowed_batch_sizes': [4, 8, 16],
        'max_enqueued_batches': 20,
        'autograph': False,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'num_batch_threads': 2,
        'max_batch_size': 32,
        'batch_timeout_micros': 2000,
        'allowed_batch_sizes': [8, 16, 32],
        'max_enqueued_batches': 25,
        'autograph': True,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'num_batch_threads': 4,
        'max_batch_size': 64,
        'batch_timeout_micros': 4000,
        'allowed_batch_sizes': [16, 32, 64],
        'max_enqueued_batches': 30,
        'autograph': False,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 128,
        'batch_timeout_micros': 8000,
        'allowed_batch_sizes': [32, 64, 128],
        'max_enqueued_batches': 35,
        'autograph': True,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'num_batch_threads': 2,
        'max_batch_size': 256,
        'batch_timeout_micros': 16000,
        'allowed_batch_sizes': [64, 128, 256],
        'max_enqueued_batches': 40,
        'autograph': False,
        'enable_large_batch_splitting': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'num_batch_threads': 4,
        'max_batch_size': 512,
        'batch_timeout_micros': 32000,
        'allowed_batch_sizes': [128, 256, 512],
        'max_enqueued_batches': 45,
        'autograph': True,
        'enable_large_batch_splitting': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 1024,
        'batch_timeout_micros': 64000,
        'allowed_batch_sizes': [256, 512, 1024],
        'max_enqueued_batches': 50,
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
