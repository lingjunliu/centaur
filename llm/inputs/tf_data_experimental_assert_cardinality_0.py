
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy


def tf_data_experimental_assert_cardinality_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.assert_cardinality.
    This API returns a transformation function for `Dataset.apply`. The testing
    framework requires the Dataset instance to be provided so it can apply the
    resulting transformation. The `dataset` key is used for this purpose.
    NOTE: tf.data.Dataset objects are not deep-copyable, so we append the dictionaries directly.
    """
    list_of_inputs = []

    # Input 1: Asserting cardinality of 0 for an empty dataset.
    input_dict_1 = {
        'dataset': tf.data.Dataset.from_tensor_slices(np.array([], dtype=np.int32)),
        'expected_cardinality': 0
    }
    list_of_inputs.append(input_dict_1)

    # Input 2: Asserting cardinality of 1 for a single-element dataset.
    input_dict_2 = {
        'dataset': tf.data.Dataset.from_tensor_slices(np.array([42], dtype=np.int32)),
        'expected_cardinality': 1
    }
    list_of_inputs.append(input_dict_2)

    # Input 3: Asserting a small positive cardinality.
    input_dict_3 = {
        'dataset': tf.data.Dataset.range(10),
        'expected_cardinality': 10
    }
    list_of_inputs.append(input_dict_3)

    # Input 4: Asserting a medium positive cardinality.
    input_dict_4 = {
        'dataset': tf.data.Dataset.from_tensor_slices(np.arange(42, dtype=np.int64)),
        'expected_cardinality': 42
    }
    list_of_inputs.append(input_dict_4)

    # Input 5: Asserting a larger positive cardinality.
    input_dict_5 = {
        'dataset': tf.data.Dataset.range(1024),
        'expected_cardinality': 1024
    }
    list_of_inputs.append(input_dict_5)

    # Input 6: Special value for infinite cardinality (-1).
    input_dict_6 = {
        'dataset': tf.data.Dataset.range(1).repeat(),
        'expected_cardinality': -1
    }
    list_of_inputs.append(input_dict_6)

    # Input 7: Special value for unknown cardinality (-2).
    input_dict_7 = {
        'dataset': tf.data.Dataset.range(20).filter(lambda x: x < 15),
        'expected_cardinality': -2
    }
    list_of_inputs.append(input_dict_7)

    # Input 8: Asserting a known cardinality on a dataset whose cardinality is statically unknown.
    input_dict_8 = {
        'dataset': tf.data.Dataset.range(50).filter(lambda x: x % 2 == 0),
        'expected_cardinality': 25
    }
    list_of_inputs.append(input_dict_8)

    # Input 9: Another small positive cardinality with a different data type.
    input_dict_9 = {
        'dataset': tf.data.Dataset.from_tensor_slices(np.linspace(0, 1, 5, dtype=np.float32)),
        'expected_cardinality': 5
    }
    list_of_inputs.append(input_dict_9)

    # Input 10: Another medium positive cardinality.
    input_dict_10 = {
        'dataset': tf.data.Dataset.range(256),
        'expected_cardinality': 256
    }
    list_of_inputs.append(input_dict_10)

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
