
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_shuffle_and_repeat_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.shuffle_and_repeat function.
    The test harness requires a key 'dataset' to provide the dataset source to apply the transformation on.
    """
    list_of_inputs = []
    
    # Define a few different dataset sources to use
    dataset_source_int = tf.constant(np.arange(50, dtype=np.int64))
    dataset_source_float = tf.constant(np.random.rand(20, 3).astype(np.float32))
    dataset_source_str = tf.constant([f"item_{i}" for i in range(30)])

    # Input 1: Basic case with integer data
    input_dict = {
        'buffer_size': np.array(20, dtype=np.int64),
        'count': np.array(2, dtype=np.int64),
        'seed': np.array(42, dtype=np.int64),
        'dataset': dataset_source_int
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Infinite repeats with float data
    input_dict = {
        'buffer_size': np.array(15, dtype=np.int64),
        'count': np.array(-1, dtype=np.int64),
        'seed': np.array(123, dtype=np.int64),
        'dataset': dataset_source_float
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Minimal buffer size with string data
    input_dict = {
        'buffer_size': np.array(1, dtype=np.int64),
        'count': np.array(3, dtype=np.int64),
        'seed': np.array(0, dtype=np.int64),
        'dataset': dataset_source_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Single epoch
    input_dict = {
        'buffer_size': np.array(30, dtype=np.int64),
        'count': np.array(1, dtype=np.int64),
        'seed': np.array(99, dtype=np.int64),
        'dataset': dataset_source_int
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative seed
    input_dict = {
        'buffer_size': np.array(25, dtype=np.int64),
        'count': np.array(4, dtype=np.int64),
        'seed': np.array(-10, dtype=np.int64),
        'dataset': dataset_source_int
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Buffer size larger than dataset (perfect shuffle)
    input_dict = {
        'buffer_size': np.array(100, dtype=np.int64),
        'count': np.array(2, dtype=np.int64),
        'seed': np.array(2023, dtype=np.int64),
        'dataset': dataset_source_int
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero count (should produce empty dataset)
    input_dict = {
        'buffer_size': np.array(10, dtype=np.int64),
        'count': np.array(0, dtype=np.int64),
        'seed': np.array(7, dtype=np.int64),
        'dataset': dataset_source_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small buffer size, large count
    input_dict = {
        'buffer_size': np.array(2, dtype=np.int64),
        'count': np.array(10, dtype=np.int64),
        'seed': np.array(1, dtype=np.int64),
        'dataset': dataset_source_int
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Another infinite repeat case with a different seed
    input_dict = {
        'buffer_size': np.array(10, dtype=np.int64),
        'count': np.array(-1, dtype=np.int64),
        'seed': np.array(1337, dtype=np.int64),
        'dataset': dataset_source_float
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Another basic case with a different seed and buffer size
    input_dict = {
        'buffer_size': np.array(29, dtype=np.int64),
        'count': np.array(3, dtype=np.int64),
        'seed': np.array(88, dtype=np.int64),
        'dataset': dataset_source_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.data.experimental.shuffle_and_repeat"] = get_shuffle_and_repeat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.shuffle_and_repeat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.shuffle_and_repeat'.")

check_valid('tf.data.experimental.shuffle_and_repeat', generated_inputs['tf.data.experimental.shuffle_and_repeat'], lib="tf", suffix=0)
