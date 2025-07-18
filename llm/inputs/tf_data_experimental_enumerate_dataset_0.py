
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_data_experimental_enumerate_dataset_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.enumerate_dataset function.
    This function returns a transformation, which needs to be applied to a dataset.
    The test harness likely expects a 'self' key containing the numpy representation of the dataset
    to which the transformation will be applied. This version uses simple numeric dtypes to
    avoid potential conversion issues in the test harness.
    """
    list_of_inputs = []

    # Input 1: Basic case with 1D integer data, start=0
    input_dict_1 = {
        'self': np.array([10, 20, 30], dtype=np.int32),
        'start': np.array(0, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Start from a positive number, with 2D float data
    input_dict_2 = {
        'self': np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32),
        'start': np.array(5, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Start from 1, with 3D integer data
    input_dict_3 = {
        'self': np.arange(8, dtype=np.int16).reshape(2, 2, 2),
        'start': np.array(1, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Large start value, with an empty dataset
    input_dict_4 = {
        'self': np.array([], dtype=np.float64),
        'start': np.array(1000000, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Negative start value, with 1D float data
    input_dict_5 = {
        'self': np.array([1.0, -2.5, 3.0], dtype=np.float32),
        'start': np.array(-1, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Larger negative start value, with a single element dataset
    input_dict_6 = {
        'self': np.array([100], dtype=np.int64),
        'start': np.array(-50, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Min int64 start value
    input_dict_7 = {
        'self': np.array([1, 2], dtype=np.int32),
        'start': np.array(np.iinfo(np.int64).min, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Max int64 start value (offset to avoid overflow during enumeration)
    input_dict_8 = {
        'self': np.array([1, 2, 3], dtype=np.int32),
        'start': np.array(np.iinfo(np.int64).max - 5, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Zero start value, different integer type
    input_dict_9 = {
        'self': np.array([5, 4, 3, 2, 1], dtype=np.uint8),
        'start': np.array(0, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Another positive start value
    input_dict_10 = {
        'self': np.array([[1], [2], [3]], dtype=np.int32),
        'start': np.array(42, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.data.experimental.enumerate_dataset"] = tf_data_experimental_enumerate_dataset_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.enumerate_dataset' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.enumerate_dataset'.")

check_valid('tf.data.experimental.enumerate_dataset', generated_inputs['tf.data.experimental.enumerate_dataset'], lib="tf", suffix=0)
