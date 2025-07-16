
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_crossed_column_inputs():
    list_of_inputs = []

    # Input 1, valid
    keys = ['feature1', 'feature2']
    hash_bucket_size = 1000
    hash_key = 'key1'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    keys = ['feature3', 'feature4', 'feature5']
    hash_bucket_size = 50000
    hash_key = 'key2'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    keys = ['feature6', 'feature7']
    hash_bucket_size = 100000
    hash_key = 'key3'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    keys = ['feature8', 'feature9']
    hash_bucket_size = 2000
    hash_key = 'key4'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    keys = ['feature10', 'feature11']
    hash_bucket_size = 75000
    hash_key = 'key5'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid with large hash_bucket_size
    keys = ['feature11', 'feature12']
    hash_bucket_size = 2147483647 # Maximum 32-bit integer
    hash_key = 'key6'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, short keys
    keys = ['a', 'b']
    hash_bucket_size = 100
    hash_key = 'key7'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid, longer keys
    keys = ['feature_very_long_1', 'feature_very_long_2']
    hash_bucket_size = 10000
    hash_key = 'key8'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid with special character in keys
    keys = ['feature!@#', 'feature$%^']
    hash_bucket_size = 20000
    hash_key = 'key9'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid with a different hash_key
    keys = ['feature1', 'feature2']
    hash_bucket_size = 1000
    hash_key = 'another_valid_key'
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
