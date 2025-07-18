
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_group_by_reducer_inputs():
    list_of_inputs = []

    # Input 1: Basic case with int32 data
    input_dict = {
        'elements': np.arange(10, dtype=np.int32),
        'key_func': [np.int64(3)],
        'reducer': [np.int32(0)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 data
    input_dict = {
        'data_stream': np.array([1.1, 2.2, 1.3, 3.1, 2.4, 3.9], dtype=np.float32),
        'key_func': [np.int64(2)],
        'reducer': [np.float32(0.0)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Dataset with negative int32 values
    input_dict = {
        'values': np.array([-1, -5, 2, -1, 5, 2, 8], dtype=np.int32),
        'key_func': [np.int64(4)],
        'reducer': [np.int32(0)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty dataset
    input_dict = {
        'empty_tensor': np.array([], dtype=np.float64),
        'key_func': [np.int64(1)],
        'reducer': [np.float64(0.0)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Dataset with a single element
    input_dict = {
        'single_element': np.array([100], dtype=np.int64),
        'key_func': [np.int64(5)],
        'reducer': [np.int64(0)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Dataset where all elements might map to the same key
    input_dict = {
        'even_numbers': np.array([2, 4, 6, 8, 10], dtype=np.int16),
        'key_func': [np.int64(2)],
        'reducer': [np.int16(0)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Dataset with unsigned integers
    input_dict = {
        'uint_data': np.arange(12, dtype=np.uint32),
        'key_func': [np.int64(4)],
        'reducer': [np.uint32(0)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger dataset with random data
    input_dict = {
        'random_ints': np.random.randint(-100, 100, size=50, dtype=np.int32),
        'key_func': [np.int64(10)],
        'reducer': [np.int32(0)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Dataset with float64 values
    input_dict = {
        'float64_data': np.linspace(-10.0, 10.0, 20, dtype=np.float64),
        'key_func': [np.int64(2)],
        'reducer': [np.float64(0.0)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Using a different modulus for the key function
    input_dict = {
        'another_tensor': np.arange(15, dtype=np.int8),
        'key_func': [np.int64(5)],
        'reducer': [np.int8(0)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.data.experimental.group_by_reducer"] = tf_data_experimental_group_by_reducer_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.group_by_reducer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.group_by_reducer'.")

check_valid('tf.data.experimental.group_by_reducer', generated_inputs['tf.data.experimental.group_by_reducer'], lib="tf", suffix=0)
