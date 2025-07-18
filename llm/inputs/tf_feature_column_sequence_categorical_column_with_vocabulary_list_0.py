
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_feature_column_sequence_categorical_column_with_vocabulary_list_inputs():
    """
    Generates a list of valid inputs for the
    tf.feature_column.sequence_categorical_column_with_vocabulary_list function.
    """
    list_of_inputs = []

    # Input 1: Basic int64 vocabulary
    input_dict_1 = {
        'key': 'product_category_int64',
        'vocabulary_list': [101, 202, 303, 404],
        'dtype': np.dtype('int64'),
        'default_value': -1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic int32 vocabulary
    input_dict_2 = {
        'key': 'user_ids_int32',
        'vocabulary_list': [1, 2, 3, 4, 5],
        'dtype': np.dtype('int32'),
        'default_value': -1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: int32 vocabulary with OOV buckets
    input_dict_3 = {
        'key': 'tags_int32_oov',
        'vocabulary_list': [10, 20, 30, 40],
        'dtype': np.dtype('int32'),
        'default_value': -1,
        'num_oov_buckets': 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: int64 vocabulary with OOV buckets
    input_dict_4 = {
        'key': 'item_codes_int64_oov',
        'vocabulary_list': [1, 2, 3, 5, 8, 13],
        'dtype': np.dtype('int64'),
        'default_value': -1,
        'num_oov_buckets': 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: int64 vocabulary with a custom default_value
    input_dict_5 = {
        'key': 'device_type_int64_default',
        'vocabulary_list': [1, 2, 3],
        'dtype': np.dtype('int64'),
        'default_value': 0,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: int32 vocabulary with a custom negative default_value
    input_dict_6 = {
        'key': 'sensor_readings_int32_default',
        'vocabulary_list': [10, 20, 30, 40, 50],
        'dtype': np.dtype('int32'),
        'default_value': -2,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Single-item int32 vocabulary with OOV
    input_dict_7 = {
        'key': 'singleton_int32_oov',
        'vocabulary_list': [42],
        'dtype': np.dtype('int32'),
        'default_value': -1,
        'num_oov_buckets': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Large int64 vocabulary list with OOV
    large_vocab = list(range(500))
    input_dict_8 = {
        'key': 'large_vocab_int64_oov',
        'vocabulary_list': large_vocab,
        'dtype': np.dtype('int64'),
        'default_value': -1,
        'num_oov_buckets': 100
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: int16 vocabulary
    input_dict_9 = {
        'key': 'small_integers_int16',
        'vocabulary_list': [0, 1, 2, 3],
        'dtype': np.dtype('int16'),
        'default_value': -1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: int32 vocabulary with negative numbers
    input_dict_10 = {
        'key': 'signed_integers_int32',
        'vocabulary_list': [-10, -5, 0, 5, 10],
        'dtype': np.dtype('int32'),
        'default_value': -99,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.feature_column.sequence_categorical_column_with_vocabulary_list"] = tf_feature_column_sequence_categorical_column_with_vocabulary_list_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.feature_column.sequence_categorical_column_with_vocabulary_list' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.sequence_categorical_column_with_vocabulary_list'.")

check_valid('tf.feature_column.sequence_categorical_column_with_vocabulary_list', generated_inputs['tf.feature_column.sequence_categorical_column_with_vocabulary_list'], lib="tf", suffix=0)
