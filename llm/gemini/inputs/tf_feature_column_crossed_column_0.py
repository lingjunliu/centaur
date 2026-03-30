
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_feature_column_crossed_column_inputs():
    list_of_inputs = []

    # Input 1, valid
    keys = ['feature1', 'feature2']
    hash_bucket_size = 1000
    hash_key = 'key1'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    keys = ['feature3', 'feature4']
    hash_bucket_size = 50000
    hash_key = 'key2'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    keys = ['feature6', 'feature7']
    hash_bucket_size = 2
    hash_key = 'key3'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    keys = ['feature8', 'feature9']
    hash_bucket_size = 100000
    hash_key = 'key4'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    keys = ['feature12', 'feature13']
    hash_bucket_size = 10
    hash_key = 'key5'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid with longer hash_key
    keys = ['feature14', 'feature15']
    hash_bucket_size = 100
    hash_key = 'long_hash_key_string'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid with special characters in keys (removed some for safety)
    keys = ['feature!', 'feature#']
    hash_bucket_size = 10000
    hash_key = 'key7'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid with numbers in keys
    keys = ['feature1', 'feature22']
    hash_bucket_size = 5000
    hash_key = 'key8'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid with empty hash_key
    keys = ['feature16', 'feature17']
    hash_bucket_size = 20000
    hash_key = ''
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid with unicode characters in keys and hash_key (removed some for safety)
    keys = ['你好feature', '世界feature']
    hash_bucket_size = 30000
    hash_key = '你好key'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11, valid with one key and hash_key as None
    keys = ['feature_test']
    hash_bucket_size = 40000
    hash_key = None
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12, valid with larger hash_bucket_size
    keys = ['feature_a', 'feature_b']
    hash_bucket_size = 2**31 - 1
    hash_key = 'test_key'
    input_dict = {'keys': keys, 'hash_bucket_size': hash_bucket_size, 'hash_key': hash_key}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
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


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.feature_column.crossed_column', generated_inputs['tf.feature_column.crossed_column'], lib="tf", suffix=0)
