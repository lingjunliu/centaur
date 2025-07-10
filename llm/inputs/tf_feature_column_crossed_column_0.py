
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_feature_column_crossed_column_inputs():
    list_of_inputs = []

    # Input 1
    keys = ['feature1', 'feature2']
    hash_bucket_size = 1000
    hash_key = 'test_key'
    input_dict = {'keys': keys, 'hash_bucket_size': np.int32(hash_bucket_size), 'hash_key': str(hash_key)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    keys = ['feature_a', 'feature_b', 'feature_c']
    hash_bucket_size = 5000
    hash_key = 'another_key'
    input_dict = {'keys': keys, 'hash_bucket_size': np.int32(hash_bucket_size), 'hash_key': str(hash_key)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    keys = ['x', 'y']
    hash_bucket_size = 10000
    hash_key = 'yet_another_key'
    input_dict = {'keys': keys, 'hash_bucket_size': np.int32(hash_bucket_size), 'hash_key': str(hash_key)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    keys = ['keyword1', 'keyword2']
    hash_bucket_size = 2000
    hash_key = 'a_new_hash_key'
    input_dict = {'keys': keys, 'hash_bucket_size': np.int32(hash_bucket_size), 'hash_key': str(hash_key)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    keys = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
    hash_bucket_size = 7500
    hash_key = 'the_latest_key'
    input_dict = {'keys': keys, 'hash_bucket_size': np.int32(hash_bucket_size), 'hash_key': str(hash_key)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    keys = ['column_a', 'column_b']
    hash_bucket_size = 1500
    hash_key = 'column_key'
    input_dict = {'keys': keys, 'hash_bucket_size': np.int32(hash_bucket_size), 'hash_key': str(hash_key)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    keys = ['val1', 'val2', 'val3']
    hash_bucket_size = 6000
    hash_key = 'test_hash'
    input_dict = {'keys': keys, 'hash_bucket_size': np.int32(hash_bucket_size), 'hash_key': str(hash_key)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    keys = ['key_one', 'key_two']
    hash_bucket_size = 3000
    hash_key = 'key_hash'
    input_dict = {'keys': keys, 'hash_bucket_size': np.int32(hash_bucket_size), 'hash_key': str(hash_key)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    keys = ['first', 'second', 'third']
    hash_bucket_size = 8000
    hash_key = 'hash_value'
    input_dict = {'keys': keys, 'hash_bucket_size': np.int32(hash_bucket_size), 'hash_key': str(hash_key)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    keys = ['primary', 'secondary']
    hash_bucket_size = 4000
    hash_key = 'hash_test'
    input_dict = {'keys': keys, 'hash_bucket_size': np.int32(hash_bucket_size), 'hash_key': str(hash_key)}
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
