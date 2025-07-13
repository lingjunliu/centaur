
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_feature_column_crossed_column_inputs():
    list_of_inputs = []

    # Input 1, valid: basic string keys
    keys = ['feature1', 'feature2']
    hash_bucket_size = 1000
    hash_key = 'key1'

    input_dict = {
        "keys": keys,
        "hash_bucket_size": hash_bucket_size,
        "hash_key": hash_key
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid: another hash key
    keys = ['feature1', 'feature2']
    hash_bucket_size = 3000
    hash_key = 'key11'

    input_dict = {
        "keys": keys,
        "hash_bucket_size": hash_bucket_size,
        "hash_key": hash_key
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid: different string hash key
    keys = ['feature1', 'feature2']
    hash_bucket_size = 7000
    hash_key = 'another_different_key'

    input_dict = {
        "keys": keys,
        "hash_bucket_size": hash_bucket_size,
        "hash_key": hash_key
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid : large number of buckets with string keys
    keys = ['feature1', 'feature2']
    hash_bucket_size = 2**20
    hash_key = 'key7'

    input_dict = {
        "keys": keys,
        "hash_bucket_size": hash_bucket_size,
        "hash_key": hash_key
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid: basic string keys different features
    keys = ['new_feature1', 'new_feature2']
    hash_bucket_size = 4000
    hash_key = 'key12'

    input_dict = {
        "keys": keys,
        "hash_bucket_size": hash_bucket_size,
        "hash_key": hash_key
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid: larger bucket size
    keys = ['featureX', 'featureY']
    hash_bucket_size = 100000
    hash_key = 'key14'

    input_dict = {
        "keys": keys,
        "hash_bucket_size": hash_bucket_size,
        "hash_key": hash_key
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid: Small bucket size
    keys = ['feature1', 'feature2']
    hash_bucket_size = 2
    hash_key = 'key10'

    input_dict = {
        "keys": keys,
        "hash_bucket_size": hash_bucket_size,
        "hash_key": hash_key
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid: string hash and different feature names
    keys = ['another_feat1', 'another_feat2']
    hash_bucket_size = 6000
    hash_key = 'string_hash2'
    
    input_dict = {
        "keys": keys,
        "hash_bucket_size": hash_bucket_size,
        "hash_key": hash_key
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid: 4 string keys
    keys = ['f1', 'f2', 'f3', 'f4']
    hash_bucket_size = 8000
    hash_key = 'key15'
    
    input_dict = {
        "keys": keys,
        "hash_bucket_size": hash_bucket_size,
        "hash_key": hash_key
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid: same key different hash bucket
    keys = ['feature1', 'feature2']
    hash_bucket_size = 5000
    hash_key = 'key1'

    input_dict = {
        "keys": keys,
        "hash_bucket_size": hash_bucket_size,
        "hash_key": hash_key
    }
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
