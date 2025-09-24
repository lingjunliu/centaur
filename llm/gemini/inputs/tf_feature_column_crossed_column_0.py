
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_feature_column_crossed_column_inputs():
    """
    Generates a list of valid inputs for tf.feature_column.crossed_column.
    The 'keys' parameter is provided as a numpy array of strings to conform to
    the "numpy format" requirement and to work with the testing harness.
    """
    list_of_inputs = []

    # Input 1: Basic case with two string keys
    input_dict_1 = {
        'keys': ['feature_a', 'feature_b'],
        'hash_bucket_size': 1000,
        'hash_key': 'key_A'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: More than two keys
    input_dict_2 = {
        'keys': ['user_id', 'product_id', 'country'],
        'hash_bucket_size': 10000,
        'hash_key': 'user_product_country_cross'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Large hash_bucket_size
    input_dict_3 = {
        'keys': ['keywords', 'doc_terms'],
        'hash_bucket_size': 50000,
        'hash_key': 'search_terms'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Minimal valid hash_bucket_size (must be > 1)
    input_dict_4 = {
        'keys': ['ad_id', 'page_id'],
        'hash_bucket_size': 2,
        'hash_key': 'minimal_bucket'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: A larger number of keys to cross
    input_dict_5 = {
        'keys': ['f1', 'f2', 'f3', 'f4', 'f5'],
        'hash_bucket_size': 100000,
        'hash_key': 'five_feature_cross'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty string as hash_key
    input_dict_6 = {
        'keys': ['city', 'state'],
        'hash_bucket_size': 500,
        'hash_key': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Long key names and a long hash_key
    input_dict_7 = {
        'keys': ['a_very_long_feature_name_for_identification', 'another_long_feature_name'],
        'hash_bucket_size': 999,
        'hash_key': 'a_very_long_and_descriptive_hash_key_for_the_crossing_operation'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Numeric characters in feature names
    input_dict_8 = {
        'keys': ['feature1', 'feature2'],
        'hash_bucket_size': 256,
        'hash_key': 'FINGERPRINT'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Prime number for hash_bucket_size
    input_dict_9 = {
        'keys': ['col_x', 'col_y', 'col_z'],
        'hash_bucket_size': 997,
        'hash_key': 'prime_bucket_size'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Single character keys
    input_dict_10 = {
        'keys': ['x', 'y'],
        'hash_bucket_size': 4096,
        'hash_key': 'xy_cross'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Keys with underscores
    input_dict_11 = {
        'keys': ['first_name', 'last_name'],
        'hash_bucket_size': 2000,
        'hash_key': 'name_cross'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))


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
