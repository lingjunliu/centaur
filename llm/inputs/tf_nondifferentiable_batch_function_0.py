
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_nondifferentiable_batch_function_inputs():
    """
    Generates a list of valid inputs for the tf.nondifferentiable_batch_function API.
    """
    list_of_inputs = []

    # Input 1: Basic case with tf.identity
    input_dict_1 = {
        'num_batch_threads': 1,
        'max_batch_size': 8,
        'batch_timeout_micros': 1000,
        'allowed_batch_sizes': [],
        'max_enqueued_batches': 10,
        'autograph': True,
        'enable_large_batch_splitting': True,
        'func': 'tf.identity',
        'inner_input_args': {
            'input': np.array([[1, 2], [3, 4]], dtype=np.float32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: With allowed_batch_sizes and tf.square
    input_dict_2 = {
        'num_batch_threads': 2,
        'max_batch_size': 32,
        'batch_timeout_micros': 2000,
        'allowed_batch_sizes': [8, 16, 32],
        'max_enqueued_batches': 20,
        'autograph': True,
        'enable_large_batch_splitting': True,
        'func': 'tf.square',
        'inner_input_args': {
            'x': np.random.rand(16, 5).astype(np.float32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Larger values, 3D tensor input and tf.add
    input_dict_3 = {
        'num_batch_threads': 8,
        'max_batch_size': 128,
        'batch_timeout_micros': 5000,
        'allowed_batch_sizes': [32, 64, 128],
        'max_enqueued_batches': 100,
        'autograph': True,
        'enable_large_batch_splitting': True,
        'func': 'tf.add',
        'inner_input_args': {
            'x': np.random.rand(64, 10, 10).astype(np.float32),
            'y': np.random.rand(64, 10, 10).astype(np.float32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Minimum valid values and integer tensor
    input_dict_4 = {
        'num_batch_threads': 1,
        'max_batch_size': 2,
        'batch_timeout_micros': 0,
        'allowed_batch_sizes': [1, 2],
        'max_enqueued_batches': 1,
        'autograph': True,
        'enable_large_batch_splitting': True,
        'func': 'tf.identity',
        'inner_input_args': {
            'input': np.array([[10], [20]], dtype=np.int32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Autograph disabled and float64 tensor
    input_dict_5 = {
        'num_batch_threads': 4,
        'max_batch_size': 64,
        'batch_timeout_micros': 10000,
        'allowed_batch_sizes': [],
        'max_enqueued_batches': 10,
        'autograph': False,
        'enable_large_batch_splitting': True,
        'func': 'tf.identity',
        'inner_input_args': {
            'input': np.random.rand(32, 1).astype(np.float64)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Large batch splitting disabled
    input_dict_6 = {
        'num_batch_threads': 4,
        'max_batch_size': 64,
        'batch_timeout_micros': 10000,
        'allowed_batch_sizes': [16, 32, 64],
        'max_enqueued_batches': 10,
        'autograph': True,
        'enable_large_batch_splitting': False,
        'func': 'tf.identity',
        'inner_input_args': {
            'input': np.random.rand(32, 100).astype(np.float32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Both boolean flags disabled and multiple inner arguments with tf.subtract
    input_dict_7 = {
        'num_batch_threads': 2,
        'max_batch_size': 16,
        'batch_timeout_micros': 500,
        'allowed_batch_sizes': [],
        'max_enqueued_batches': 5,
        'autograph': False,
        'enable_large_batch_splitting': False,
        'func': 'tf.subtract',
        'inner_input_args': {
            'x': np.ones((8, 4), dtype=np.float32),
            'y': np.zeros((8, 4), dtype=np.float32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 4D tensor and complex allowed_batch_sizes
    input_dict_8 = {
        'num_batch_threads': 16,
        'max_batch_size': 256,
        'batch_timeout_micros': 20000,
        'allowed_batch_sizes': [8, 16, 32, 64, 128, 256],
        'max_enqueued_batches': 50,
        'autograph': True,
        'enable_large_batch_splitting': True,
        'func': 'tf.identity',
        'inner_input_args': {
            'input': np.zeros((128, 2, 2, 3), dtype=np.float32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: High max_enqueued_batches and int64 tensor with tf.negative
    input_dict_9 = {
        'num_batch_threads': 4,
        'max_batch_size': 32,
        'batch_timeout_micros': 3000,
        'allowed_batch_sizes': [8, 16, 32],
        'max_enqueued_batches': 1000,
        'autograph': True,
        'enable_large_batch_splitting': True,
        'func': 'tf.negative',
        'inner_input_args': {
            'x': np.random.randint(0, 100, size=(10, 20)).astype(np.int64)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Long timeout and boolean tensor with tf.logical_not
    input_dict_10 = {
        'num_batch_threads': 1,
        'max_batch_size': 10,
        'batch_timeout_micros': 1000000,
        'allowed_batch_sizes': [5, 10],
        'max_enqueued_batches': 10,
        'autograph': False,
        'enable_large_batch_splitting': False,
        'func': 'tf.logical_not',
        'inner_input_args': {
            'x': np.array([True, False, True, False, True]).reshape(5, 1).astype(bool)
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
