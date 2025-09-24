
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_prefetch_to_device_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.prefetch_to_device.
    The API returns a transformation function. The test harness expects the 'self' key
    for the initial dataset and the function's arguments to be nested under an 'args' key.
    """
    list_of_inputs = []

    # Input 1: Basic case with CPU, int32 data, and a specified buffer size.
    list_of_inputs.append(copy.deepcopy({
        'self': np.array([1, 2, 3, 4, 5], dtype=np.int32),
        'args': {
            'device': '/cpu:0',
            'buffer_size': 2
        }
    }))

    # Input 2: GPU target, float32 data, with default buffer size (omitted).
    list_of_inputs.append(copy.deepcopy({
        'self': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'args': {
            'device': '/gpu:0'
        }
    }))

    # Input 3: Dataset composed of a tuple of numpy arrays.
    list_of_inputs.append(copy.deepcopy({
        'self': (np.array([1, 2, 3], dtype=np.int32), np.array([4.0, 5.0, 6.0], dtype=np.float32)),
        'args': {
            'device': '/gpu:0',
            'buffer_size': 4
        }
    }))

    # Input 4: String data on CPU with a small buffer.
    list_of_inputs.append(copy.deepcopy({
        'self': np.array(["alpha", "beta", "gamma"], dtype=str),
        'args': {
            'device': '/cpu:0',
            'buffer_size': 1
        }
    }))

    # Input 5: Edge case with an empty dataset.
    list_of_inputs.append(copy.deepcopy({
        'self': np.array([], dtype=np.int64),
        'args': {
            'device': '/gpu:0',
            'buffer_size': 10
        }
    }))

    # Input 6: High-dimensional data with a full device string.
    list_of_inputs.append(copy.deepcopy({
        'self': np.random.rand(2, 3, 4).astype(np.float64),
        'args': {
            'device': '/job:localhost/replica:0/task:0/device:CPU:0',
            'buffer_size': 5
        }
    }))

    # Input 7: Boolean data.
    list_of_inputs.append(copy.deepcopy({
        'self': np.array([True, False, True, False], dtype=np.bool_),
        'args': {
            'device': '/cpu:0',
            'buffer_size': 3
        }
    }))

    # Input 8: Unsigned integer data type.
    list_of_inputs.append(copy.deepcopy({
        'self': np.arange(10, dtype=np.uint8),
        'args': {
            'device': '/gpu:0',
            'buffer_size': 8
        }
    }))
    
    # Input 9: Zero buffer size.
    list_of_inputs.append(copy.deepcopy({
        'self': np.array([10, 20, 30], dtype=np.int32),
        'args': {
            'device': '/cpu:0',
            'buffer_size': 0
        }
    }))

    # Input 10: Another GPU index with a larger buffer.
    list_of_inputs.append(copy.deepcopy({
        'self': np.ones((5, 5), dtype=np.int16),
        'args': {
            'device': '/gpu:1',
            'buffer_size': 16
        }
    }))

    return list_of_inputs

generated_inputs["tf.data.experimental.prefetch_to_device"] = get_prefetch_to_device_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.prefetch_to_device' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.prefetch_to_device'.")

check_valid('tf.data.experimental.prefetch_to_device', generated_inputs['tf.data.experimental.prefetch_to_device'], lib="tf", suffix=0)
