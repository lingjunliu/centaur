
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_arange_inputs():
    """
    This function generates a list of valid inputs for the
    tf.experimental.numpy.arange function.
    """
    list_of_inputs = []

    # Input 1: Basic integer range
    input_dict_1 = {
        'start': np.array(0, dtype=np.int32),
        'stop': np.array(10, dtype=np.int32),
        'step': np.array(1, dtype=np.int32),
        'dtype': np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Integer range with a different step
    input_dict_2 = {
        'start': np.array(0, dtype=np.int64),
        'stop': np.array(20, dtype=np.int64),
        'step': np.array(2, dtype=np.int64),
        'dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Negative start value
    input_dict_3 = {
        'start': np.array(-5, dtype=np.int32),
        'stop': np.array(5, dtype=np.int32),
        'step': np.array(1, dtype=np.int32),
        'dtype': np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Negative step value (counting down)
    input_dict_4 = {
        'start': np.array(10, dtype=np.int32),
        'stop': np.array(-10, dtype=np.int32),
        'step': np.array(-2, dtype=np.int32),
        'dtype': np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Basic float range with float32 dtype
    input_dict_5 = {
        'start': np.array(0.0, dtype=np.float32),
        'stop': np.array(1.0, dtype=np.float32),
        'step': np.array(0.1, dtype=np.float32),
        'dtype': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Float range with float64 dtype
    input_dict_6 = {
        'start': np.array(0.0, dtype=np.float64),
        'stop': np.array(5.0, dtype=np.float64),
        'step': np.array(0.5, dtype=np.float64),
        'dtype': np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty range where start equals stop
    input_dict_7 = {
        'start': np.array(5, dtype=np.int32),
        'stop': np.array(5, dtype=np.int32),
        'step': np.array(1, dtype=np.int32),
        'dtype': np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Empty range where start > stop with positive step
    input_dict_8 = {
        'start': np.array(10, dtype=np.int32),
        'stop': np.array(0, dtype=np.int32),
        'step': np.array(1, dtype=np.int32),
        'dtype': np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Large numbers with int64
    input_dict_9 = {
        'start': np.array(1000000, dtype=np.int64),
        'stop': np.array(1000010, dtype=np.int64),
        'step': np.array(1, dtype=np.int64),
        'dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Mixed integer and float values, with consistent float types
    input_dict_10 = {
        'start': np.array(0.0, dtype=np.float64),
        'stop': np.array(5.5, dtype=np.float64),
        'step': np.array(1.0, dtype=np.float64),
        'dtype': np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Empty range where start < stop with negative step
    input_dict_11 = {
        'start': np.array(-5, dtype=np.int32),
        'stop': np.array(5, dtype=np.int32),
        'step': np.array(-1, dtype=np.int32),
        'dtype': np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    # Input 12: Range that includes only the start value
    input_dict_12 = {
        'start': np.array(4.5, dtype=np.float32),
        'stop': np.array(5.0, dtype=np.float32),
        'step': np.array(1.0, dtype=np.float32),
        'dtype': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.arange"] = tf_experimental_numpy_arange_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.arange' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.arange'.")

check_valid('tf.experimental.numpy.arange', generated_inputs['tf.experimental.numpy.arange'], lib="tf", suffix=0)
