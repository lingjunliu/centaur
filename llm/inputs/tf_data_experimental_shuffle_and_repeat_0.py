
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_data_experimental_shuffle_and_repeat_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.shuffle_and_repeat function.
    This function returns a transformation function. The testing framework expects the data
    (from which a dataset will be created) to be provided under the 'inner_values' key,
    and the function's arguments to be nested under a 'kwargs' key.
    """
    list_of_inputs = []

    # Input 1: Basic case with a specific seed
    input_dict_1 = {
        'inner_values': np.arange(200, dtype=np.int64),
        'kwargs': {
            'buffer_size': np.array(100, dtype=np.int64),
            'count': np.array(2, dtype=np.int64),
            'seed': np.array(42, dtype=np.int64)
        }
    }
    list_of_inputs.append(input_dict_1)

    # Input 2: Minimal buffer_size (1)
    input_dict_2 = {
        'inner_values': np.arange(10, dtype=np.int32),
        'kwargs': {
            'buffer_size': np.array(1, dtype=np.int64),
            'count': np.array(5, dtype=np.int64),
            'seed': np.array(1, dtype=np.int64)
        }
    }
    list_of_inputs.append(input_dict_2)

    # Input 3: Indefinite repeat (count=-1) with float data
    input_dict_3 = {
        'inner_values': np.arange(50, dtype=np.float32).reshape(10, 5),
        'kwargs': {
            'buffer_size': np.array(20, dtype=np.int64),
            'count': np.array(-1, dtype=np.int64),
            'seed': np.array(2023, dtype=np.int64)
        }
    }
    list_of_inputs.append(input_dict_3)

    # Input 4: Large buffer_size (perfect shuffling) and count, seed is 0
    input_dict_4 = {
        'inner_values': np.random.rand(100, 2).astype(np.float64),
        'kwargs': {
            'buffer_size': np.array(100, dtype=np.int64),
            'count': np.array(10, dtype=np.int64),
            'seed': np.array(0, dtype=np.int64)
        }
    }
    list_of_inputs.append(input_dict_4)

    # Input 5: Negative seed
    input_dict_5 = {
        'inner_values': np.arange(100, dtype=np.uint8),
        'kwargs': {
            'buffer_size': np.array(50, dtype=np.int64),
            'count': np.array(3, dtype=np.int64),
            'seed': np.array(-10, dtype=np.int64)
        }
    }
    list_of_inputs.append(input_dict_5)

    # Input 6: Large seed value
    input_dict_6 = {
        'inner_values': np.arange(500, dtype=np.int16),
        'kwargs': {
            'buffer_size': np.array(256, dtype=np.int64),
            'count': np.array(4, dtype=np.int64),
            'seed': np.array(987654321, dtype=np.int64)
        }
    }
    list_of_inputs.append(input_dict_6)

    # Input 7: Count is 1 (repeat once)
    input_dict_7 = {
        'inner_values': np.linspace(0, 1, 200, dtype=np.float16),
        'kwargs': {
            'buffer_size': np.array(128, dtype=np.int64),
            'count': np.array(1, dtype=np.int64),
            'seed': np.array(55, dtype=np.int64)
        }
    }
    list_of_inputs.append(input_dict_7)

    # Input 8: Another indefinite repeat case with negative seed
    input_dict_8 = {
        'inner_values': np.arange(-500, 500, dtype=np.int64),
        'kwargs': {
            'buffer_size': np.array(500, dtype=np.int64),
            'count': np.array(-1, dtype=np.int64),
            'seed': np.array(-1, dtype=np.int64)
        }
    }
    list_of_inputs.append(input_dict_8)

    # Input 9: Small buffer_size and large count with boolean data
    input_dict_9 = {
        'inner_values': np.array([True, False] * 10),
        'kwargs': {
            'buffer_size': np.array(2, dtype=np.int64),
            'count': np.array(100, dtype=np.int64),
            'seed': np.array(123, dtype=np.int64)
        }
    }
    list_of_inputs.append(input_dict_9)

    # Input 10: Values from the documentation example (with a seed)
    input_dict_10 = {
        'inner_values': np.array([1, 2, 3], dtype=np.int32),
        'kwargs': {
            'buffer_size': np.array(2, dtype=np.int64),
            'count': np.array(2, dtype=np.int64),
            'seed': np.array(1337, dtype=np.int64)
        }
    }
    list_of_inputs.append(input_dict_10)

    return list_of_inputs

generated_inputs["tf.data.experimental.shuffle_and_repeat"] = tf_data_experimental_shuffle_and_repeat_inputs()

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
