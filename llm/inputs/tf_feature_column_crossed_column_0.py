
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
    keys = ['feature_a', 'feature_b', 'feature_c']
    hash_bucket_size = 50000
    hash_key = 'key2'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    keys = ['feature_x', 'feature_y']
    hash_bucket_size = 100000
    hash_key = 'key3'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    keys = ['f1', 'f2']
    hash_bucket_size = 1000000
    hash_key = 'key4'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    keys = ['feature_8', 'feature_9']
    hash_bucket_size = 5000
    hash_key = 'key5'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    keys = ['feature_12', 'feature_13']
    hash_bucket_size = 75000
    hash_key = 'different_key'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    keys = ['feature_14', 'feature_15']
    hash_bucket_size = 2
    hash_key = 'key7'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    keys = ['feature_16', 'feature_17']
    hash_bucket_size = 20000
    hash_key = 'key8'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, Valid
    keys = ['name_1', 'name_2']
    hash_bucket_size = 15000
    hash_key = 'string_key'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, Valid
    keys = ['f_1','f_2']
    hash_bucket_size = 30000
    hash_key = 'key_10'
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
