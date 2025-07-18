
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_assert_cardinality_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.assert_cardinality.
    This function returns a transformation function. To make it testable, the inputs
    include a pickleable representation of a dataset (under the 'dataset' key)
    which the test harness is expected to use to apply the transformation to.
    """
    list_of_inputs = []

    # Input 1: Correct cardinality for a simple dataset of integers.
    input_dict_1 = {
        'expected_cardinality': 10,
        'dataset': np.arange(10, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Correct cardinality for an empty dataset.
    input_dict_2 = {
        'expected_cardinality': 0,
        'dataset': np.array([], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Correct cardinality for a dataset with a single, multi-dimensional element.
    input_dict_3 = {
        'expected_cardinality': 1,
        'dataset': np.random.rand(1, 5, 5).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Using a numpy integer type for the cardinality argument.
    input_dict_4 = {
        'expected_cardinality': np.int32(25),
        'dataset': np.arange(25)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Dataset constructed from a tuple of numpy arrays.
    input_dict_5 = {
        'expected_cardinality': 8,
        'dataset': (np.arange(8), np.linspace(0, 1, 8, dtype=np.float32))
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Dataset constructed from a dictionary of numpy arrays.
    input_dict_6 = {
        'expected_cardinality': 4,
        'dataset': {'features': np.random.rand(4, 16), 'labels': np.arange(4)}
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Dataset with string elements.
    input_dict_7 = {
        'expected_cardinality': 3,
        'dataset': np.array(['cat', 'dog', 'mouse'], dtype=object)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Asserting infinite cardinality.
    # The test harness must create a .repeat() dataset to make this pass.
    input_dict_8 = {
        'expected_cardinality': tf.data.experimental.INFINITE_CARDINALITY,
        'dataset': np.arange(5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Asserting unknown cardinality.
    # The test harness must create a .filter() dataset to make this pass.
    input_dict_9 = {
        'expected_cardinality': tf.data.experimental.UNKNOWN_CARDINALITY,
        'dataset': np.arange(10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: A valid API call that should produce a runtime error because the cardinality is wrong.
    input_dict_10 = {
        'expected_cardinality': 99,
        'dataset': np.arange(100)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.data.experimental.assert_cardinality"] = tf_data_experimental_assert_cardinality_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.assert_cardinality' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.assert_cardinality'.")

check_valid('tf.data.experimental.assert_cardinality', generated_inputs['tf.data.experimental.assert_cardinality'], lib="tf", suffix=0)
