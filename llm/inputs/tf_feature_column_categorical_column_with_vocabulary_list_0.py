
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def categorical_column_with_vocabulary_list_inputs():
    """
    Generates a list of valid inputs for tf.feature_column.categorical_column_with_vocabulary_list.
    This version corrects the dtype mismatch by using np.dtype() objects for the dtype parameter.
    It continues to use only integer vocabularies to avoid issues with external tools applying numerical
    operations on the list.
    """
    list_of_inputs = []

    # Input 1: Basic int64 vocabulary with num_oov_buckets, using np.dtype
    input_dict_1 = {
        'key': 'item_ids',
        'vocabulary_list': [10, 20, 30, 40],
        'dtype': np.dtype('int64'),
        'default_value': -1,
        'num_oov_buckets': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic int64 vocabulary with a non-default default_value, using np.dtype
    input_dict_2 = {
        'key': 'category_codes',
        'vocabulary_list': [101, 102, 103],
        'dtype': np.dtype('int64'),
        'default_value': 0,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Vocabulary with dtype=None (to be inferred as int64)
    input_dict_3 = {
        'key': 'group_ids',
        'vocabulary_list': [1, 2, 3, 5, 8],
        'dtype': None,
        'default_value': -1,
        'num_oov_buckets': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Correctly specified int32 vocabulary, using np.dtype
    input_dict_4 = {
        'key': 'status_codes',
        'vocabulary_list': [np.int32(200), np.int32(404), np.int32(500)],
        'dtype': np.dtype('int32'),
        'default_value': -1,
        'num_oov_buckets': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Single-item int64 vocabulary list, using np.dtype
    input_dict_5 = {
        'key': 'singleton_feature',
        'vocabulary_list': [999],
        'dtype': np.dtype('int64'),
        'default_value': -1,
        'num_oov_buckets': 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: default_value is a value outside the vocab index range
    input_dict_6 = {
        'key': 'class_indices',
        'vocabulary_list': [0, 1, 2],
        'dtype': None,
        'default_value': 10,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Large num_oov_buckets, using np.dtype
    input_dict_7 = {
        'key': 'hashed_feature',
        'vocabulary_list': [1001, 1002],
        'dtype': np.dtype('int64'),
        'default_value': -1,
        'num_oov_buckets': 1000
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Vocabulary with negative integers, using np.dtype
    input_dict_8 = {
        'key': 'offset_values',
        'vocabulary_list': [-10, -5, 0, 5, 10],
        'dtype': np.dtype('int64'),
        'default_value': -1,
        'num_oov_buckets': 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Longer vocabulary list with a default_value and inferred dtype
    input_dict_9 = {
        'key': 'sequential_ids',
        'vocabulary_list': list(range(10)),
        'dtype': None,
        'default_value': 0,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Non-sequential int32 vocabulary, using np.dtype
    input_dict_10 = {
        'key': 'zip_codes',
        'vocabulary_list': [np.int32(90210), np.int32(10001), np.int32(60606)],
        'dtype': np.dtype('int32'),
        'default_value': -1,
        'num_oov_buckets': 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: A different negative default_value
    input_dict_11 = {
        'key': 'error_codes',
        'vocabulary_list': [404, 500, 503],
        'dtype': None,
        'default_value': -2,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    # Input 12: Larger list with explicit np.int32 and matching np.dtype
    input_dict_12 = {
        'key': 'sensor_readings',
        'vocabulary_list': [np.int32(x) for x in range(20)],
        'dtype': np.dtype('int32'),
        'default_value': -1,
        'num_oov_buckets': 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))
    
    return list_of_inputs

generated_inputs["tf.feature_column.categorical_column_with_vocabulary_list"] = categorical_column_with_vocabulary_list_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.feature_column.categorical_column_with_vocabulary_list' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.categorical_column_with_vocabulary_list'.")

check_valid('tf.feature_column.categorical_column_with_vocabulary_list', generated_inputs['tf.feature_column.categorical_column_with_vocabulary_list'], lib="tf", suffix=0)
