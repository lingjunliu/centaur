
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_crossed_column_inputs():
    list_of_inputs = []

    # Input 1: Basic string keys
    keys = ['feature1', 'feature2']
    hash_bucket_size = 1000
    hash_key = 'test_key'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: String keys with larger bucket size
    keys = ['feature_a', 'feature_b']
    hash_bucket_size = 50000
    hash_key = 'another_key'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Two string keys
    keys = ['string_col_1', 'string_col_2']
    hash_bucket_size = 100
    hash_key = 'key_3'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Short key
    keys = ['a', 'b']
    hash_bucket_size = 200
    hash_key = 'key_4'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: String keys, different hash key
    keys = ['feature_x', 'feature_y']
    hash_bucket_size = 1000
    hash_key = 'another_test_key'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger number of buckets, short hash key
    keys = ['k1', 'k2']
    hash_bucket_size = 100000
    hash_key = 'a'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small number of buckets
    keys = ['f1', 'f2']
    hash_bucket_size = 2
    hash_key = 'key7'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Two string keys, numerical hash key
    keys = ['col1', 'col2']
    hash_bucket_size = 500
    hash_key = '12345'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Two string keys, long hash key
    keys = ['string_a', 'string_b']
    hash_bucket_size = 750
    hash_key = 'very_long_hash_key_for_testing'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: String keys with numbers
    keys = ['feature_1', 'feature_2']
    hash_bucket_size = 25000
    hash_key = 'numeric_key'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.crossed_column"] = tf_feature_column_crossed_column_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.crossed_column' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.crossed_column'.")

check_valid('tf.feature_column.crossed_column', generated_inputs['tf.feature_column.crossed_column'], lib="tf", suffix=0)
