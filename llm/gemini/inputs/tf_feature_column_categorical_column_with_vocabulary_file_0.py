
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_categorical_column_with_vocabulary_file_inputs():
    list_of_inputs = []

    # Input 1: Basic string vocabulary with default value
    input_dict = {
        'key': 'color',
        'vocabulary_file': 'colors.txt',
        'vocabulary_size': 3,
        'dtype': tf.string,
        'default_value': -1,
        'num_oov_buckets': 0,
        'file_format': 'text'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic integer vocabulary with oov buckets
    input_dict = {
        'key': 'number',
        'vocabulary_file': 'numbers.txt',
        'vocabulary_size': 5,
        'dtype': tf.int64,
        'default_value': None,
        'num_oov_buckets': 2,
        'file_format': 'text'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: String vocabulary, no default, no oov
    input_dict = {
        'key': 'fruit',
        'vocabulary_file': 'fruits.txt',
        'vocabulary_size': 4,
        'dtype': tf.string,
        'default_value': None,
        'num_oov_buckets': 0,
        'file_format': 'text'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Integer vocabulary, default 0
    input_dict = {
        'key': 'age',
        'vocabulary_file': 'ages.txt',
        'vocabulary_size': 10,
        'dtype': tf.int64,
        'default_value': 0,
        'num_oov_buckets': 0,
        'file_format': 'text'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small string vocabulary with file format
    input_dict = {
        'key': 'city',
        'vocabulary_file': 'cities.tfrecord.gz',
        'vocabulary_size': 2,
        'dtype': tf.string,
        'default_value': -1,
        'num_oov_buckets': 0,
        'file_format': 'tfrecord_gzip'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer with oov
    input_dict = {
        'key': 'id',
        'vocabulary_file': 'ids.txt',
        'vocabulary_size': 7,
        'dtype': tf.int64,
        'default_value': None,
        'num_oov_buckets': 3,
        'file_format': 'text'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger vocabulary
    input_dict = {
        'key': 'product',
        'vocabulary_file': 'products.txt',
        'vocabulary_size': 50,
        'dtype': tf.string,
        'default_value': -1,
        'num_oov_buckets': 0,
        'file_format': 'text'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: Integer vocabulary with oov buckets, smaller vocabulary size
    input_dict = {
        'key': 'number_small',
        'vocabulary_file': 'numbers_small.txt',
        'vocabulary_size': 3,
        'dtype': tf.int64,
        'default_value': None,
        'num_oov_buckets': 1,
        'file_format': 'text'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: string, no default, file format explicit
    input_dict = {
        'key': 'animal',
        'vocabulary_file': 'animals.txt',
        'vocabulary_size': 6,
        'dtype': tf.string,
        'default_value': None,
        'num_oov_buckets': 0,
        'file_format': 'text'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10:  Integer vocabulary, default -1
    input_dict = {
        'key': 'rank',
        'vocabulary_file': 'ranks.txt',
        'vocabulary_size': 15,
        'dtype': tf.int64,
        'default_value': -1,
        'num_oov_buckets': 0,
        'file_format': 'text'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.categorical_column_with_vocabulary_file"] = tf_feature_column_categorical_column_with_vocabulary_file_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.categorical_column_with_vocabulary_file' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.categorical_column_with_vocabulary_file'.")

check_valid('tf.feature_column.categorical_column_with_vocabulary_file', generated_inputs['tf.feature_column.categorical_column_with_vocabulary_file'], lib="tf", suffix=0)
