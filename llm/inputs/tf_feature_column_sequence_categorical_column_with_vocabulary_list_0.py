
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_sequence_categorical_column_with_vocabulary_list_inputs():
    list_of_inputs = []

    # Input 1: Simple string vocabulary
    input_dict = {
        'key': 'colors',
        'vocabulary_list': ['R', 'G', 'B'],
        'dtype': np.dtype(np.str_),
        'default_value': -1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer vocabulary
    input_dict = {
        'key': 'numbers',
        'vocabulary_list': [1, 2, 3, 4, 5],
        'dtype': np.dtype(np.int64),
        'default_value': 0,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: String vocabulary with OOV buckets
    input_dict = {
        'key': 'cities',
        'vocabulary_list': ['London', 'Paris', 'Tokyo'],
        'dtype': np.dtype(np.str_),
        'default_value': -1,
        'num_oov_buckets': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Integer vocabulary with default value
    input_dict = {
        'key': 'ages',
        'vocabulary_list': [18, 21, 25, 30],
        'dtype': np.dtype(np.int32),
        'default_value': -2,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Mixed string vocabulary - REMOVING '123' to ensure all elements are string
    input_dict = {
        'key': 'items',
        'vocabulary_list': ['apple', 'banana', 'cherry'],
        'dtype': np.dtype(np.str_),
        'default_value': -1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger number of OOV buckets
    input_dict = {
        'key': 'large_oov',
        'vocabulary_list': ['A', 'B'],
        'dtype': np.dtype(np.str_),
        'default_value': -1,
        'num_oov_buckets': 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative numbers in vocabulary
    input_dict = {
        'key': 'negative_numbers',
        'vocabulary_list': [-1, -2, -3, 0, 1],
        'dtype': np.dtype(np.int32),
        'default_value': -4,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: More vocabulary items
    input_dict = {
        'key': 'many_items',
        'vocabulary_list': [str(i) for i in range(100)],
        'dtype': np.dtype(np.str_),
        'default_value': -1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Different default value
    input_dict = {
        'key': 'default_value_test',
        'vocabulary_list': ['x', 'y', 'z'],
        'dtype': np.dtype(np.str_),
        'default_value': 10,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Different dtype (int16)
    input_dict = {
        'key': 'int16_numbers',
        'vocabulary_list': [1, 2, 3],
        'dtype': np.dtype(np.int16),
        'default_value': 0,
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
