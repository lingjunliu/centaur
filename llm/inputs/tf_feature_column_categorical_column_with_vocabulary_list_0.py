
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_categorical_column_with_vocabulary_list_inputs():
    list_of_inputs = []

    # Input 1: String vocabulary, default default_value
    input_dict = {
        'key': 'color',
        'vocabulary_list': ['red', 'green', 'blue'],
        'dtype': tf.string,
        'default_value': -1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer vocabulary, num_oov_buckets
    input_dict = {
        'key': 'number',
        'vocabulary_list': [1, 2, 3, 4],
        'dtype': tf.int64,
        'default_value': -1,
        'num_oov_buckets': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: String vocabulary, num_oov_buckets
    input_dict = {
        'key': 'city',
        'vocabulary_list': ['london', 'paris', 'tokyo'],
        'dtype': tf.string,
        'default_value': -1,
        'num_oov_buckets': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Integer vocabulary, default default_value
    input_dict = {
        'key': 'age',
        'vocabulary_list': [20, 30, 40, 50],
        'dtype': tf.int32,
        'default_value': 0,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: String vocabulary with special characters
    input_dict = {
        'key': 'item',
        'vocabulary_list': ['item1', 'item_2', 'item-3'],
        'dtype': tf.string,
        'default_value': -1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.categorical_column_with_vocabulary_list"] = tf_feature_column_categorical_column_with_vocabulary_list_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.categorical_column_with_vocabulary_list' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.categorical_column_with_vocabulary_list'.")

check_valid('tf.feature_column.categorical_column_with_vocabulary_list', generated_inputs['tf.feature_column.categorical_column_with_vocabulary_list'], lib="tf", suffix=0)
