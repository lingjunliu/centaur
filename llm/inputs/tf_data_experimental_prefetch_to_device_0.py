
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy


def tf_data_experimental_prefetch_to_device_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.prefetch_to_device.
    For transformation functions that are applied to a dataset, the test harness
    expects the source data for the dataset under the 'input' key.
    """
    list_of_inputs = []

    # Input 1: Basic prefetch to CPU with a small integer dataset
    input_dict_1 = {
        'input': np.arange(10, dtype=np.int32),
        'device': '/cpu:0',
        'buffer_size': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Prefetch to GPU with a float dataset
    input_dict_2 = {
        'input': np.arange(20, dtype=np.float32).reshape(5, 4),
        'device': '/gpu:0',
        'buffer_size': 8
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Prefetch with a larger buffer size
    input_dict_3 = {
        'input': np.random.rand(50, 2).astype(np.float32),
        'device': '/cpu:0',
        'buffer_size': 16
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Prefetching a dataset of dictionaries
    input_dict_4 = {
        'input': {
            'a': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            'b': np.array([6, 7, 8, 9, 10], dtype=np.int32)
        },
        'device': '/gpu:0',
        'buffer_size': 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Prefetching a dataset of tuples
    input_dict_5 = {
        'input': (np.arange(5, dtype=np.int32), np.arange(5, 10, dtype=np.float64)),
        'device': '/cpu:0',
        'buffer_size': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Full device string for CPU
    input_dict_6 = {
        'input': np.array([True, False, True]),
        'device': '/job:localhost/replica:0/task:0/device:CPU:0',
        'buffer_size': 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Full device string for GPU
    input_dict_7 = {
        'input': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'device': '/job:localhost/replica:0/task:0/device:GPU:0',
        'buffer_size': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Prefetching a dataset of strings (cannot be placed on GPU)
    input_dict_8 = {
        'input': np.array(['apple', 'banana', 'cherry']),
        'device': '/cpu:0',
        'buffer_size': 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Empty dataset
    input_dict_9 = {
        'input': np.array([], dtype=np.int32),
        'device': '/cpu:0',
        'buffer_size': 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Larger dataset with a specific buffer
    input_dict_10 = {
        'input': np.arange(1000, dtype=np.int64),
        'device': '/gpu:0',
        'buffer_size': 128
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.data.experimental.prefetch_to_device"] = tf_data_experimental_prefetch_to_device_inputs()

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
