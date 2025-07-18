
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_crossed_column_inputs():
    list_of_inputs = []

    # The API's internal check `if not keys:` causes a `ValueError` when `keys` is a
    # NumPy array, as the truthiness of a multi-element array is ambiguous.
    # To fix this, `keys` must be provided as a standard Python list, which is
    # consistent with the API documentation and the specified signature {'keys': 'list'}.

    # Input 1: Simple case with two string keys
    input_dict_1 = {
        'keys': ['feature_a', 'feature_b'],
        'hash_bucket_size': 1000,
        'hash_key': 'KEY_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: More than two string keys
    input_dict_2 = {
        'keys': ['user_gender', 'user_country', 'ad_category'],
        'hash_bucket_size': 50000,
        'hash_key': 'CROSS_KEY_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Minimal valid hash_bucket_size
    input_dict_3 = {
        'keys': ['brand', 'model'],
        'hash_bucket_size': 2,
        'hash_key': 'MIN_BUCKET'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Long feature names
    input_dict_4 = {
        'keys': ['a_very_long_feature_name_that_is_still_valid', 'another_similarly_long_feature_name'],
        'hash_bucket_size': 10000,
        'hash_key': 'LONG_KEYS'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Large number of keys
    input_dict_5 = {
        'keys': ['key1', 'key2', 'key3', 'key4', 'key5'],
        'hash_bucket_size': 100000,
        'hash_key': 'MULTI_KEY'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty string for hash_key
    input_dict_6 = {
        'keys': ['city', 'state'],
        'hash_bucket_size': 1000000,
        'hash_key': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Single character keys
    input_dict_7 = {
        'keys': ['x', 'y', 'z'],
        'hash_bucket_size': 100,
        'hash_key': 'XYZ_CROSS'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Numeric-like string keys
    input_dict_8 = {
        'keys': ['2023', '11', '15'],
        'hash_bucket_size': 500,
        'hash_key': 'DATE_CROSS'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Mix of short and long keys
    input_dict_9 = {
        'keys': ['short_key', 'a_much_longer_key_for_testing_purposes'],
        'hash_bucket_size': 25000,
        'hash_key': 'MIXED_LENGTH_KEYS'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Very long hash key
    input_dict_10 = {
        'keys': ['product_id', 'session_id'],
        'hash_bucket_size': 75000,
        'hash_key': 'a_very_long_and_specific_hash_key_used_for_fingerprinting_in_this_specific_crossing_scenario'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.feature_column.crossed_column"] = tf_feature_column_crossed_column_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.feature_column.crossed_column' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.crossed_column'.")

check_valid('tf.feature_column.crossed_column', generated_inputs['tf.feature_column.crossed_column'], lib="tf", suffix=0)
