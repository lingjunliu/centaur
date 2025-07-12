
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_feature_column_indicator_column_inputs():
    list_of_inputs = []

    # Input 1:  categorical_column_with_vocabulary_list
    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_list(
        'color', ['red', 'green', 'blue'])]
    input_dict = {'categorical_column': categorical_column}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: categorical_column_with_identity
    categorical_column = [tf.feature_column.categorical_column_with_identity(
        key='user_id', num_buckets=5)]
    input_dict = {'categorical_column': categorical_column}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: crossed_column
    feature_columns = [
        tf.feature_column.categorical_column_with_vocabulary_list(
            'color', vocabulary_list=('red', 'green', 'blue')),
        tf.feature_column.categorical_column_with_vocabulary_list(
            'size', vocabulary_list=('S', 'M', 'L'))
    ]
    categorical_column = [tf.feature_column.crossed_column(
        feature_columns, hash_bucket_size=10)]
    input_dict = {'categorical_column': categorical_column}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: categorical_column_with_hash_bucket
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=1000)]
    input_dict = {'categorical_column': categorical_column}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple categorical columns in a list (single column)
    categorical_column = [
        tf.feature_column.categorical_column_with_vocabulary_list(
            'city', ['New York', 'London', 'Paris'])
    ]
    input_dict = {'categorical_column': categorical_column}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6:  Different type of vocabulary list
    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_list(
        'number', ['1', '2', '3'])]
    input_dict = {'categorical_column': categorical_column}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Crossed column without hash bucket
    feature_columns = [
        tf.feature_column.categorical_column_with_vocabulary_list(
            'color', vocabulary_list=('red', 'green', 'blue')),
        tf.feature_column.categorical_column_with_vocabulary_list(
            'size', vocabulary_list=('S', 'M', 'L')),
    ]
    categorical_column = [tf.feature_column.crossed_column(
        feature_columns, hash_bucket_size=1000)]
    input_dict = {'categorical_column': categorical_column}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large identity bucket
    categorical_column = [tf.feature_column.categorical_column_with_identity(
        key='item_id', num_buckets=10000)]
    input_dict = {'categorical_column': categorical_column}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: numerical values with identity
    categorical_column = [tf.feature_column.categorical_column_with_identity(
        key='price', num_buckets=100)]
    input_dict = {'categorical_column': categorical_column}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Crossed column with single vocabulary
    feature_columns = [
        tf.feature_column.categorical_column_with_vocabulary_list(
            'feature1', vocabulary_list=['a']),
        tf.feature_column.categorical_column_with_vocabulary_list(
            'feature2', vocabulary_list=['b'])
    ]
    categorical_column = [tf.feature_column.crossed_column(
        feature_columns, hash_bucket_size=100)]
    input_dict = {'categorical_column': categorical_column}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.indicator_column"] = tf_feature_column_indicator_column_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.indicator_column' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.indicator_column'.")

check_valid('tf.feature_column.indicator_column', generated_inputs['tf.feature_column.indicator_column'], lib="tf", suffix=0)
