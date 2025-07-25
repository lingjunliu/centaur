
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf


def decorated_func_add(x):
    return x + x

def decorated_func_square(a):
    return tf.square(a)

def decorated_func_add_two_args(x, y):
    return x + y


def tf_nondifferentiable_batch_function_inputs():
    list_of_inputs = []

    # Input 1: Basic case with large batch splitting disabled
    input_dict_1 = {
        'num_batch_threads': 1,
        'max_batch_size': 32,
        'batch_timeout_micros': 10000,
        'allowed_batch_sizes': [],
        'max_enqueued_batches': 10,
        'autograph': True,
        'enable_large_batch_splitting': False,
        'fn': decorated_func_add,
        'kwargs': {
            'x': np.random.rand(16, 8).astype(np.float32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Large batch splitting enabled, input size > max_batch_size
    input_dict_2 = {
        'num_batch_threads': 4,
        'max_batch_size': 64,
        'batch_timeout_micros': 20000,
        'allowed_batch_sizes': [],
        'max_enqueued_batches': 20,
        'autograph': False,
        'enable_large_batch_splitting': True,
        'fn': decorated_func_square,
        'kwargs': {
            'a': np.random.rand(100, 5).astype(np.float32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: With allowed_batch_sizes, splitting disabled
    input_dict_3 = {
        'num_batch_threads': 2,
        'max_batch_size': 16,
        'batch_timeout_micros': 5000,
        'allowed_batch_sizes': [8, 16],
        'max_enqueued_batches': 10,
        'autograph': True,
        'enable_large_batch_splitting': False,
        'fn': decorated_func_add,
        'kwargs': {
            'x': np.random.rand(12, 3).astype(np.float32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Inner function with multiple arguments, splitting enabled
    input_dict_4 = {
        'num_batch_threads': 2,
        'max_batch_size': 16,
        'batch_timeout_micros': 1000,
        'allowed_batch_sizes': [],
        'max_enqueued_batches': 5,
        'autograph': True,
        'enable_large_batch_splitting': True,
        'fn': decorated_func_add_two_args,
        'kwargs': {
            'x': np.ones((20, 1)).astype(np.float32),
            'y': np.ones((20, 1)).astype(np.float32) * 2
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: High number of threads, splitting disabled
    input_dict_5 = {
        'num_batch_threads': 16,
        'max_batch_size': 256,
        'batch_timeout_micros': 50000,
        'allowed_batch_sizes': [],
        'max_enqueued_batches': 100,
        'autograph': True,
        'enable_large_batch_splitting': False,
        'fn': decorated_func_square,
        'kwargs': {
            'a': np.random.rand(200, 10).astype(np.float32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Minimal allowed_batch_sizes list, splitting enabled
    input_dict_6 = {
        'num_batch_threads': 2,
        'max_batch_size': 32,
        'batch_timeout_micros': 8000,
        'allowed_batch_sizes': [32],
        'max_enqueued_batches': 10,
        'autograph': True,
        'enable_large_batch_splitting': True,
        'fn': decorated_func_add,
        'kwargs': {
            'x': np.random.rand(64, 2).astype(np.float32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Zero timeout, splitting disabled
    input_dict_7 = {
        'num_batch_threads': 1,
        'max_batch_size': 64,
        'batch_timeout_micros': 0,
        'allowed_batch_sizes': [],
        'max_enqueued_batches': 10,
        'autograph': True,
        'enable_large_batch_splitting': False,
        'fn': decorated_func_square,
        'kwargs': {
            'a': np.random.rand(1, 16).astype(np.float32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Larger values for integer parameters, splitting enabled
    input_dict_8 = {
        'num_batch_threads': 32,
        'max_batch_size': 512,
        'batch_timeout_micros': 100000,
        'allowed_batch_sizes': [64, 128, 256, 512],
        'max_enqueued_batches': 50,
        'autograph': False,
        'enable_large_batch_splitting': True,
        'fn': decorated_func_add_two_args,
        'kwargs': {
            'x': np.random.rand(1024, 20).astype(np.float32),
            'y': np.random.rand(1024, 20).astype(np.float32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Non-power-of-two batch sizes, splitting disabled
    input_dict_9 = {
        'num_batch_threads': 3,
        'max_batch_size': 90,
        'batch_timeout_micros': 12345,
        'allowed_batch_sizes': [30, 60, 90],
        'max_enqueued_batches': 12,
        'autograph': False,
        'enable_large_batch_splitting': False,
        'fn': decorated_func_add,
        'kwargs': {
            'x': np.random.rand(85, 1).astype(np.float32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Splitting enabled but input size is smaller
    input_dict_10 = {
        'num_batch_threads': 1,
        'max_batch_size': 128,
        'batch_timeout_micros': 15000,
        'allowed_batch_sizes': [],
        'max_enqueued_batches': 10,
        'autograph': True,
        'enable_large_batch_splitting': True,
        'fn': decorated_func_square,
        'kwargs': {
            'a': np.random.rand(40, 4).astype(np.float32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

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

check_valid('tf.nondifferentiable_batch_function', generated_inputs['tf.nondifferentiable_batch_function'], lib="tf", suffix=0)
