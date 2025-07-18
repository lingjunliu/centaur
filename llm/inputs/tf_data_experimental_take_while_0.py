
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_take_while_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.take_while function.
    The inputs are structured to be compatible with a test harness that expects:
    1. A dataset to be provided under the key 'input'.
    2. The 'predicate' argument to be a list, as per the strict signature requirement.
    """
    list_of_inputs = []

    # The error "the input does not have inner values" suggests the test harness
    # needs a specific key for the dataset itself. Previous attempts with 'dataset'
    # and 'self' failed. Trying the generic key 'input'.

    # Input 1: Basic integer dataset
    input_dict_1 = {
        'input': np.arange(10, dtype=np.int32),
        'predicate': [True, True, True, True, True, False, True, True, True, True]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Float dataset
    input_dict_2 = {
        'input': np.array([1.1, 2.2, 3.3, -1.0, 4.4], dtype=np.float64),
        'predicate': [True, True, False, True, True]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Predicate causes taking zero elements
    input_dict_3 = {
        'input': np.arange(5, dtype=np.int64),
        'predicate': [False, True, True, True, True]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Predicate causes taking all elements
    input_dict_4 = {
        'input': np.array([10, 20, 30], dtype=np.int32),
        'predicate': [True, True, True]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Empty input dataset
    input_dict_5 = {
        'input': np.array([], dtype=np.float32),
        'predicate': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Dataset with vector elements
    input_dict_6 = {
        'input': np.arange(12, dtype=np.int32).reshape(4, 3),
        'predicate': [True, True, False, True]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Dataset with negative numbers
    input_dict_7 = {
        'input': np.array([-2, -1, 0, 1, 2], dtype=np.int32),
        'predicate': [True, True, True, False, True]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Predicate list is shorter than the dataset
    input_dict_8 = {
        'input': np.array([1, 2, 3, 4], dtype=np.int32),
        'predicate': [True, True]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Non-empty dataset, empty predicate list
    input_dict_9 = {
        'input': np.array([5, 5, 5], dtype=np.int32),
        'predicate': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Dataset with 2D tensor elements
    input_dict_10 = {
        'input': np.ones((5, 2, 3), dtype=np.float32),
        'predicate': [True, True, True, True, False]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.data.experimental.take_while"] = tf_data_experimental_take_while_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.take_while' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.take_while'.")

check_valid('tf.data.experimental.take_while', generated_inputs['tf.data.experimental.take_while'], lib="tf", suffix=0)
