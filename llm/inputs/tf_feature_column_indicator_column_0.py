
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_indicator_column_inputs():
    list_of_inputs = []

    # Input 1: Simple vocabulary list
    categorical_column1 = tf.feature_column.categorical_column_with_vocabulary_list(
        'feature1', vocabulary_list=['a', 'b', 'c'])
    input_dict1 = {'categorical_column': [categorical_column1]}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Vocabulary file
    categorical_column2 = tf.feature_column.categorical_column_with_vocabulary_file(
        'feature2', vocabulary_file='vocabulary.txt', vocabulary_size=3)
    with open('vocabulary.txt', 'w') as f:
        f.write('x\ny\nz')
    input_dict2 = {'categorical_column': [categorical_column2]}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Identity
    categorical_column3 = tf.feature_column.categorical_column_with_identity(
        'feature3', num_buckets=5)
    input_dict3 = {'categorical_column': [categorical_column3]}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Hash bucket
    categorical_column4 = tf.feature_column.categorical_column_with_hash_bucket(
        'feature4', hash_bucket_size=10)
    input_dict4 = {'categorical_column': [categorical_column4]}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Weighted categorical column
    categorical_column5 = tf.feature_column.categorical_column_with_vocabulary_list(
        'feature5', vocabulary_list=['p', 'q', 'r'])
    weighted_column = tf.feature_column.weighted_categorical_column(
        categorical_column5, weight_feature_key='weights')
    input_dict5 = {'categorical_column': [weighted_column]}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Crossed column. Taking only valid categorical columns for crossing
    categorical_column6_1 = tf.feature_column.categorical_column_with_vocabulary_list(
        'feature6_1', vocabulary_list=['u', 'v'])
    categorical_column6_2 = tf.feature_column.categorical_column_with_vocabulary_list(
        'feature6_2', vocabulary_list=['w', 'x'])
    crossed_column = tf.feature_column.crossed_column(
        [categorical_column6_1, categorical_column6_2], hash_bucket_size=15)
    input_dict6 = {'categorical_column': [crossed_column]}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7:  Taking only one feature column
    categorical_column7_1 = tf.feature_column.categorical_column_with_vocabulary_list(
        'feature7_1', vocabulary_list=['s', 't'])
    input_dict7 = {'categorical_column': [categorical_column7_1]}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Taking only vocabulary list for crossed column as other types are giving errors
    categorical_column8_1 = tf.feature_column.categorical_column_with_vocabulary_list(
        'feature8_1', vocabulary_list=['alpha', 'beta'])
    categorical_column8_2 = tf.feature_column.categorical_column_with_vocabulary_list(
        'feature8_2', vocabulary_list=['gamma', 'delta'])
    crossed_column2 = tf.feature_column.crossed_column(
        [categorical_column8_1, categorical_column8_2], hash_bucket_size=20)
    input_dict8 = {'categorical_column': [crossed_column2]}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 10: Larger number of buckets. Using vocabulary list instead of identity
    categorical_column10 = tf.feature_column.categorical_column_with_vocabulary_list(
        'feature10', vocabulary_list=[str(i) for i in range(100)])
    input_dict10 = {'categorical_column': [categorical_column10]}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11: Add a valid input after removing the invalid one from Input 9
    categorical_column11 = tf.feature_column.categorical_column_with_vocabulary_list(
            'feature11', vocabulary_list=['d', 'e'])
    input_dict11 = {'categorical_column': [categorical_column11]}
    list_of_inputs.append(copy.deepcopy(input_dict11))


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
