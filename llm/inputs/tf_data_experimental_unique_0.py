
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_unique_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.unique API.
    This API returns a transformation function. The test harness expects the data for the
    initial dataset to be provided in a nested dictionary under the 'inner_values' key.
    The 'inner_values' dictionary should contain arguments for tf.data.Dataset.from_tensor_slices,
    primarily the 'tensors' key. The tf.data.experimental.unique function itself takes no
    arguments, so the top-level dictionary outside of 'inner_values' is empty.
    """
    list_of_inputs = []

    # Input 1: Simple 1D array with integer duplicates
    input_dict = {
        'inner_values': {
            'tensors': np.array([1, 37, 2, 37, 2, 1], dtype=np.int32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array with all unique elements
    input_dict = {
        'inner_values': {
            'tensors': np.array([10, 20, 30, 40, 50], dtype=np.int64)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array with all elements being the same
    input_dict = {
        'inner_values': {
            'tensors': np.array([5, 5, 5, 5, 5], dtype=np.int32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array with negative numbers, zero, and duplicates
    input_dict = {
        'inner_values': {
            'tensors': np.array([-1, 0, 1, -1, 0, -2], dtype=np.int32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array of floating-point numbers with duplicates
    input_dict = {
        'inner_values': {
            'tensors': np.array([1.1, 2.2, 1.1, 3.3, 2.2], dtype=np.float32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array of strings with duplicates
    input_dict = {
        'inner_values': {
            'tensors': np.array(['apple', 'banana', 'cherry', 'apple', 'banana'], dtype=object)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array with duplicate rows
    input_dict = {
        'inner_values': {
            'tensors': np.array([[1, 2], [3, 4], [1, 2], [5, 6]], dtype=np.int32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array with all unique rows
    input_dict = {
        'inner_values': {
            'tensors': np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int32)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty array
    input_dict = {
        'inner_values': {
            'tensors': np.array([], dtype=np.float64)
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array of booleans with duplicates
    input_dict = {
        'inner_values': {
            'tensors': np.array([True, False, True, True, False])
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.data.experimental.unique"] = tf_data_experimental_unique_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.unique' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.unique'.")

check_valid('tf.data.experimental.unique', generated_inputs['tf.data.experimental.unique'], lib="tf", suffix=0)
