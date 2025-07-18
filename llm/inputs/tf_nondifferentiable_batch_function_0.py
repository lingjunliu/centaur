
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def get_tf_nondifferentiable_batch_function_inputs():
    """
    Generates a list of valid inputs for the tf.nondifferentiable_batch_function decorator.
    This includes 'inputs' as a key for the decorated function's call arguments.
    """
    list_of_inputs = []

    # Input 1: Basic case with a single tensor argument for the decorated function.
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 16,
        'batch_timeout_micros': 10000,
        'allowed_batch_sizes': [],
        'max_enqueued_batches': 10,
        'autograph': True,
        'enable_large_batch_splitting': True,
        'inputs': (np.random.rand(8, 4).astype(np.float32),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Using allowed_batch_sizes. The input batch size of 10 will be padded to 16.
    input_dict = {
        'num_batch_threads': 2,
        'max_batch_size': 32,
        'batch_timeout_micros': 20000,
        'allowed_batch_sizes': [16, 32],
        'max_enqueued_batches': 20,
        'autograph': True,
        'enable_large_batch_splitting': True,
        'inputs': (np.random.rand(10, 5).astype(np.float32),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: autograph and enable_large_batch_splitting are False.
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 8,
        'batch_timeout_micros': 5000,
        'allowed_batch_sizes': [],
        'max_enqueued_batches': 10,
        'autograph': False,
        'enable_large_batch_splitting': False,
        'inputs': (np.random.rand(4, 2).astype(np.float64),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Decorated function with two tensor arguments.
    input_dict = {
        'num_batch_threads': 4,
        'max_batch_size': 64,
        'batch_timeout_micros': 15000,
        'allowed_batch_sizes': [32, 64],
        'max_enqueued_batches': 10,
        'autograph': True,
        'enable_large_batch_splitting': True,
        'inputs': (
            np.random.randint(0, 100, size=(20, 10), dtype=np.int32),
            np.random.randint(0, 100, size=(20, 5), dtype=np.int32)
        )
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger values for parameters and larger inner tensor.
    input_dict = {
        'num_batch_threads': 8,
        'max_batch_size': 128,
        'batch_timeout_micros': 100000,
        'allowed_batch_sizes': [64, 128],
        'max_enqueued_batches': 50,
        'autograph': True,
        'enable_large_batch_splitting': True,
        'inputs': (np.ones((100, 10), dtype=np.int64),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Minimal valid values.
    input_dict = {
        'num_batch_threads': 1,
        'max_batch_size': 1,
        'batch_timeout_micros': 0,
        'allowed_batch_sizes': [1],
        'max_enqueued_batches': 1,
        'autograph': False,
        'enable_large_batch_splitting': False,
        'inputs': (np.random.rand(1, 1).astype(np.float32),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nondifferentiable_batch_function"] = get_tf_nondifferentiable_batch_function_inputs()

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
