
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_unique_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.unique.
    This API returns a function to be used with `Dataset.apply`. The test harness
    is expected to create a dataset from the numpy array provided under the 'dataset' key.
    The signature for `tf.data.experimental.unique()` itself is `{}`, so no direct
    arguments are passed to it.
    """
    list_of_inputs = []

    # Case 1: Basic case with integers and duplicates
    input_dict_1 = {
        'dataset': np.array([1, 37, 2, 37, 2, 1], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: All elements are unique
    input_dict_2 = {
        'dataset': np.array([1, 2, 3, 4, 5], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: All elements are the same
    input_dict_3 = {
        'dataset': np.array([5, 5, 5, 5, 5], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Empty input
    input_dict_4 = {
        'dataset': np.array([], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Negative numbers and zero
    input_dict_5 = {
        'dataset': np.array([-1, -2, -1, 0, 5, 0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Floating point numbers
    input_dict_6 = {
        'dataset': np.array([1.1, 2.2, 1.1, 3.3, -4.4, 2.2], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Strings
    input_dict_7 = {
        'dataset': np.array(["hello", "world", "hello", "tensorflow", "world"], dtype=np.object_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: 2D arrays as elements
    input_dict_8 = {
        'dataset': np.array([[1, 2], [3, 4], [1, 2]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: Boolean values
    input_dict_9 = {
        'dataset': np.array([True, False, True, True, False], dtype=np.bool_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: Single element
    input_dict_10 = {
        'dataset': np.array([100], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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
