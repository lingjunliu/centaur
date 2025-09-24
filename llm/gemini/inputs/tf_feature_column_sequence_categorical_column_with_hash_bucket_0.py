
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_sequence_categorical_column_with_hash_bucket_inputs():
    list_of_inputs = []

    # Input 1, valid
    key = "tokens1"
    hash_bucket_size = 100
    dtype = tf.string

    input_dict = {
        "key": key,
        "hash_bucket_size": hash_bucket_size,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    key = "tokens2"
    hash_bucket_size = 1000
    dtype = tf.int64

    input_dict = {
        "key": key,
        "hash_bucket_size": hash_bucket_size,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    key = "ids3"
    hash_bucket_size = 2
    dtype = tf.int32

    input_dict = {
        "key": key,
        "hash_bucket_size": hash_bucket_size,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    key = "words4"
    hash_bucket_size = 10000
    dtype = tf.string

    input_dict = {
        "key": key,
        "hash_bucket_size": hash_bucket_size,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    key = "indices5"
    hash_bucket_size = 5
    dtype = tf.int16

    input_dict = {
        "key": key,
        "hash_bucket_size": hash_bucket_size,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    key = "features6"
    hash_bucket_size = 100000
    dtype = tf.string

    input_dict = {
        "key": key,
        "hash_bucket_size": hash_bucket_size,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    key = "values7"
    hash_bucket_size = 20
    dtype = tf.int8

    input_dict = {
        "key": key,
        "hash_bucket_size": hash_bucket_size,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    key = "data8"
    hash_bucket_size = 123
    dtype = tf.string

    input_dict = {
        "key": key,
        "hash_bucket_size": hash_bucket_size,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
        
    # Input 9, valid
    key = "sequence9"
    hash_bucket_size = 987
    dtype = tf.int32

    input_dict = {
        "key": key,
        "hash_bucket_size": hash_bucket_size,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    key = "timestamps10"
    hash_bucket_size = 2345
    dtype = tf.int64

    input_dict = {
        "key": key,
        "hash_bucket_size": hash_bucket_size,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.sequence_categorical_column_with_hash_bucket"] = tf_feature_column_sequence_categorical_column_with_hash_bucket_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.sequence_categorical_column_with_hash_bucket' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.sequence_categorical_column_with_hash_bucket'.")

check_valid('tf.feature_column.sequence_categorical_column_with_hash_bucket', generated_inputs['tf.feature_column.sequence_categorical_column_with_hash_bucket'], lib="tf", suffix=0)
