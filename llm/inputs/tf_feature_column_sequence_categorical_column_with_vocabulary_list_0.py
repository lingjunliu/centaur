
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np

def tf_feature_column_sequence_categorical_column_with_vocabulary_list_inputs():
    list_of_inputs = []

    # Input 1, valid
    key = 'color'
    vocabulary_list = ['red', 'green', 'blue']
    dtype = np.dtype(np.str_)
    default_value = -1
    num_oov_buckets = 0

    input_dict = {
        "key": key,
        "vocabulary_list": vocabulary_list,
        "dtype": dtype,
        "default_value": default_value,
        "num_oov_buckets": num_oov_buckets
    }
    list_of_inputs.append(input_dict)

    # Input 2, valid
    key = 'number'
    vocabulary_list = [1, 2, 3]
    dtype = np.dtype(np.int64)
    default_value = 0
    num_oov_buckets = 0

    input_dict = {
        "key": key,
        "vocabulary_list": vocabulary_list,
        "dtype": dtype,
        "default_value": default_value,
        "num_oov_buckets": num_oov_buckets
    }
    list_of_inputs.append(input_dict)

    # Input 3, valid, with oov buckets
    key = 'city'
    vocabulary_list = ['london', 'paris', 'tokyo']
    dtype = np.dtype(np.str_)
    default_value = -1
    num_oov_buckets = 2

    input_dict = {
        "key": key,
        "vocabulary_list": vocabulary_list,
        "dtype": dtype,
        "default_value": default_value,
        "num_oov_buckets": num_oov_buckets
    }
    list_of_inputs.append(input_dict)

    # Input 4, valid, with different default value
    key = 'shape'
    vocabulary_list = ['square', 'circle', 'triangle']
    dtype = np.dtype(np.str_)
    default_value = 1
    num_oov_buckets = 0

    input_dict = {
        "key": key,
        "vocabulary_list": vocabulary_list,
        "dtype": dtype,
        "default_value": default_value,
        "num_oov_buckets": num_oov_buckets
    }
    list_of_inputs.append(input_dict)

     # Input 5, integer vocabs with negative values
    key = 'integer_values'
    vocabulary_list = [-1, 0, 1, 2]
    dtype = np.dtype(np.int64)
    default_value = -2
    num_oov_buckets = 0

    input_dict = {
        "key": key,
        "vocabulary_list": vocabulary_list,
        "dtype": dtype,
        "default_value": default_value,
        "num_oov_buckets": num_oov_buckets
    }
    list_of_inputs.append(input_dict)

    # Input 6, Large number of OOV buckets
    key = 'location'
    vocabulary_list = ['us', 'uk', 'ca']
    dtype = np.dtype(np.str_)
    default_value = -1
    num_oov_buckets = 100

    input_dict = {
        "key": key,
        "vocabulary_list": vocabulary_list,
        "dtype": dtype,
        "default_value": default_value,
        "num_oov_buckets": num_oov_buckets
    }
    list_of_inputs.append(input_dict)

    # Input 7, dtype is int32
    key = 'number_32'
    vocabulary_list = [1, 2, 3]
    dtype = np.dtype(np.int32)
    default_value = 0
    num_oov_buckets = 0

    input_dict = {
        "key": key,
        "vocabulary_list": vocabulary_list,
        "dtype": dtype,
        "default_value": default_value,
        "num_oov_buckets": num_oov_buckets
    }
    list_of_inputs.append(input_dict)

    # Input 8, Large default value
    key = 'large_number'
    vocabulary_list = [1, 2, 3]
    dtype = np.dtype(np.int64)
    default_value = 10000
    num_oov_buckets = 0

    input_dict = {
        "key": key,
        "vocabulary_list": vocabulary_list,
        "dtype": dtype,
        "default_value": default_value,
        "num_oov_buckets": num_oov_buckets
    }
    list_of_inputs.append(input_dict)
    
    # Input 9, limited vocabulary list with oov buckets and string dtype
    key = 'limited_strings'
    vocabulary_list = ['apple', 'banana']
    dtype = np.dtype(np.str_)
    default_value = -1
    num_oov_buckets = 1

    input_dict = {
        "key": key,
        "vocabulary_list": vocabulary_list,
        "dtype": dtype,
        "default_value": default_value,
        "num_oov_buckets": num_oov_buckets
    }
    list_of_inputs.append(input_dict)
    
    # Input 10: Different default value with oov_buckets, remove negative value for default value
    key = 'diff_default_oov'
    vocabulary_list = ['cat', 'dog']
    dtype = np.dtype(np.str_)
    default_value = 0
    num_oov_buckets = 2
    
    input_dict = {
        "key": key,
        "vocabulary_list": vocabulary_list,
        "dtype": dtype,
        "default_value": default_value,
        "num_oov_buckets": num_oov_buckets
    }
    list_of_inputs.append(input_dict)
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.sequence_categorical_column_with_vocabulary_list"] = tf_feature_column_sequence_categorical_column_with_vocabulary_list_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.sequence_categorical_column_with_vocabulary_list' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.sequence_categorical_column_with_vocabulary_list'.")

check_valid('tf.feature_column.sequence_categorical_column_with_vocabulary_list', generated_inputs['tf.feature_column.sequence_categorical_column_with_vocabulary_list'], lib="tf", suffix=0)
