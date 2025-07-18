
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def get_tf_data_experimental_shuffle_and_repeat_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.shuffle_and_repeat.
    This version omits optional arguments (count, seed) in some cases to allow
    the testing framework to use the function's default values. It uses the
    'inner_values' key for the dataset as suggested by the error message.
    """
    list_of_inputs = []

    # Input 1: Only the required argument 'buffer_size'.
    # count and seed will be default (None).
    input_dict = {
        'inner_values': np.arange(100, dtype=np.int32),
        'buffer_size': np.array(50, dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Provide 'count', let 'seed' be default.
    input_dict = {
        'inner_values': np.random.rand(80, 2).astype(np.float32),
        'buffer_size': np.array(80, dtype=np.int64),
        'count': np.array(3, dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Provide all arguments, including 'seed'.
    input_dict = {
        'inner_values': np.arange(200, dtype=np.int64),
        'buffer_size': np.array(100, dtype=np.int64),
        'count': np.array(2, dtype=np.int64),
        'seed': np.array(42, dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Indefinite repeat (count=-1).
    input_dict = {
        'inner_values': np.arange(150, dtype=np.int32),
        'buffer_size': np.array(75, dtype=np.int64),
        'count': np.array(-1, dtype=np.int64),
        'seed': np.array(123, dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Minimal buffer size, no seed.
    input_dict = {
        'inner_values': np.arange(30, dtype=np.float64),
        'buffer_size': np.array(1, dtype=np.int64),
        'count': np.array(5, dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Repeat only once, perfect shuffle.
    input_dict = {
        'inner_values': np.arange(10, dtype=np.uint8),
        'buffer_size': np.array(10, dtype=np.int64),
        'count': np.array(1, dtype=np.int64),
        'seed': np.array(99, dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Only buffer_size, larger dataset.
    input_dict = {
        'inner_values': np.arange(1000, dtype=np.int32),
        'buffer_size': np.array(500, dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: All args, zero seed.
    input_dict = {
        'inner_values': np.arange(50, dtype=np.int32),
        'buffer_size': np.array(25, dtype=np.int64),
        'count': np.array(4, dtype=np.int64),
        'seed': np.array(0, dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Provide count, no seed, indefinite repeat.
    input_dict = {
        'inner_values': np.arange(60, dtype=np.int32),
        'buffer_size': np.array(30, dtype=np.int64),
        'count': np.array(-1, dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: All args, large seed.
    input_dict = {
        'inner_values': np.arange(500, dtype=np.int32),
        'buffer_size': np.array(256, dtype=np.int64),
        'count': np.array(2, dtype=np.int64),
        'seed': np.array(1337, dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.data.experimental.shuffle_and_repeat"] = get_tf_data_experimental_shuffle_and_repeat_inputs()

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
