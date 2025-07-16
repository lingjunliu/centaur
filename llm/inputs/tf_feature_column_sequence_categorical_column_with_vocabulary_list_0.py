
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_sequence_categorical_column_with_vocabulary_list_inputs():
    list_of_inputs = []

    # Input 1: Basic string vocabulary
    input_dict = {
        'key': 'color',
        'vocabulary_list': ['red', 'green', 'blue'],
        'dtype': np.dtype('object'),
        'default_value': -1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer vocabulary
    input_dict = {
        'key': 'number',
        'vocabulary_list': [1, 2, 3, 4, 5],
        'dtype': np.dtype('int64'),
        'default_value': 0,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: String vocabulary with OOV buckets
    input_dict = {
        'key': 'city',
        'vocabulary_list': ['london', 'paris', 'tokyo'],
        'dtype': np.dtype('object'),
        'default_value': -1,
        'num_oov_buckets': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Integer vocabulary with default value
    input_dict = {
        'key': 'age',
        'vocabulary_list': [18, 21, 25, 30],
        'dtype': np.dtype('int32'),
        'default_value': -2,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty vocabulary_list (should raise error but test input)
    input_dict = {
        'key': 'empty',
        'vocabulary_list': [],
        'dtype': np.dtype('object'),
        'default_value': -1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: String vocabulary with mixed case
    input_dict = {
        'key': 'fruit',
        'vocabulary_list': ['Apple', 'banana', 'ORANGE'],
        'dtype': np.dtype('object'),
        'default_value': -1,
        'num_oov_buckets': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Integer vocabulary with negative values
    input_dict = {
        'key': 'score',
        'vocabulary_list': [-10, -5, 0, 5, 10],
        'dtype': np.dtype('int64'),
        'default_value': 0,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: String vocabulary with special characters
    input_dict = {
        'key': 'symbol',
        'vocabulary_list': ['!', '@', '#', '$', '%'],
        'dtype': np.dtype('object'),
        'default_value': -1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: Integer vocabulary, large numbers
    input_dict = {
        'key': 'large_number',
        'vocabulary_list': [1000000, 2000000, 3000000],
        'dtype': np.dtype('int64'),
        'default_value': 0,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Dtype inferred from list of strings
    input_dict = {
        'key': 'profession',
        'vocabulary_list': ['doctor', 'engineer', 'teacher'],
        'dtype': None,
        'default_value': -1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: String Vocabulary with numpy strings
    input_dict = {
        'key': 'location',
        'vocabulary_list': [np.array('USA'), np.array('Canada'), np.array('Mexico')],
        'dtype': np.dtype('object'),
        'default_value': -1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
