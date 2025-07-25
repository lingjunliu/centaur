
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_copy_to_device_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.copy_to_device.
    The test harness requires a special 'inner_values' key containing a tuple
    of numpy arrays to create the initial dataset for the transformation.
    """
    list_of_inputs = []

    # Input 1: Basic CPU to GPU copy with float32 data
    input_dict_1 = {
        'target_device': '/gpu:0',
        'source_device': '/cpu:0',
        'inner_values': (np.random.rand(10, 5).astype(np.float32),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic GPU to CPU copy with int32 data
    input_dict_2 = {
        'target_device': '/cpu:0',
        'source_device': '/gpu:0',
        'inner_values': (np.random.randint(0, 100, size=(8,), dtype=np.int32),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Shorthand device names with boolean data
    input_dict_3 = {
        'target_device': 'GPU:0',
        'source_device': 'CPU:0',
        'inner_values': (np.array([True, False, True, False, True], dtype=np.bool_),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: CPU to CPU copy with float16 data
    input_dict_4 = {
        'target_device': '/cpu:0',
        'source_device': '/cpu:0',
        'inner_values': (np.random.rand(2, 3, 4).astype(np.float16),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: GPU to GPU copy with float64 data
    input_dict_5 = {
        'target_device': '/gpu:0',
        'source_device': '/gpu:0',
        'inner_values': (np.random.rand(5).astype(np.float64),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Using canonical device names with uint8 data
    input_dict_6 = {
        'target_device': '/device:GPU:0',
        'source_device': '/device:CPU:0',
        'inner_values': (np.arange(10, dtype=np.uint8),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Complex numbers (complex64)
    input_dict_7 = {
        'target_device': '/gpu:0',
        'source_device': '/cpu:0',
        'inner_values': (np.array([1+2j, 3-4j, 5+0j], dtype=np.complex64),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Another integer type (int16) with default source device
    input_dict_8 = {
        'target_device': '/gpu:0',
        'inner_values': (np.array([-100, 0, 100, 1000], dtype=np.int16),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Using full device path with zero-initialized data
    input_dict_9 = {
        'target_device': '/job:localhost/replica:0/task:0/device:GPU:0',
        'source_device': '/job:localhost/replica:0/task:0/device:CPU:0',
        'inner_values': (np.zeros((3, 3), dtype=np.float32),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Single element dataset
    input_dict_10 = {
        'target_device': '/gpu:0',
        'source_device': '/cpu:0',
        'inner_values': (np.array([42], dtype=np.int64),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.data.experimental.copy_to_device"] = tf_data_experimental_copy_to_device_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.copy_to_device' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.copy_to_device'.")

check_valid('tf.data.experimental.copy_to_device', generated_inputs['tf.data.experimental.copy_to_device'], lib="tf", suffix=0)
