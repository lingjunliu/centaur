
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_indicator_column_inputs():
    list_of_inputs = []

    # Using categorical_column_with_vocabulary_list
    # Input 1: Basic string vocabulary
    cat_col_1 = tf.feature_column.categorical_column_with_vocabulary_list(
        'feature1', vocabulary_list=['a', 'b', 'c'])
    list_of_inputs.append(copy.deepcopy({'categorical_column': cat_col_1}))

    # Input 2: Numeric vocabulary (as strings)
    cat_col_2 = tf.feature_column.categorical_column_with_vocabulary_list(
        'feature2', vocabulary_list=['101', '202', '303'])
    list_of_inputs.append(copy.deepcopy({'categorical_column': cat_col_2}))

    # Input 3: With OOV (out-of-vocabulary) buckets
    cat_col_3 = tf.feature_column.categorical_column_with_vocabulary_list(
        'feature3', vocabulary_list=['cat', 'dog'], num_oov_buckets=1)
    list_of_inputs.append(copy.deepcopy({'categorical_column': cat_col_3}))

    # Input 4: With integer vocabulary and integer dtype
    cat_col_4 = tf.feature_column.categorical_column_with_vocabulary_list(
        'feature4', vocabulary_list=[10, 20, 30], dtype=tf.int64)
    list_of_inputs.append(copy.deepcopy({'categorical_column': cat_col_4}))
    
    # Input 5: With a default value
    cat_col_5 = tf.feature_column.categorical_column_with_vocabulary_list(
        'feature5', vocabulary_list=['X', 'Y'], default_value=0)
    list_of_inputs.append(copy.deepcopy({'categorical_column': cat_col_5}))

    # Input 6: With multiple OOV buckets
    cat_col_6 = tf.feature_column.categorical_column_with_vocabulary_list(
        'feature6', vocabulary_list=['apple', 'banana'], num_oov_buckets=5)
    list_of_inputs.append(copy.deepcopy({'categorical_column': cat_col_6}))

    # Using categorical_column_with_hash_bucket
    # Input 7: Basic hash bucket
    cat_col_7 = tf.feature_column.categorical_column_with_hash_bucket(
        'feature7', hash_bucket_size=100)
    list_of_inputs.append(copy.deepcopy({'categorical_column': cat_col_7}))

    # Input 8: Hash bucket with int64 dtype
    cat_col_8 = tf.feature_column.categorical_column_with_hash_bucket(
        'feature8', hash_bucket_size=50, dtype=tf.int64)
    list_of_inputs.append(copy.deepcopy({'categorical_column': cat_col_8}))

    # Input 9: Large hash bucket size
    cat_col_9 = tf.feature_column.categorical_column_with_hash_bucket(
        'feature9', hash_bucket_size=10000)
    list_of_inputs.append(copy.deepcopy({'categorical_column': cat_col_9}))

    # Input 10: Small hash bucket size
    cat_col_10 = tf.feature_column.categorical_column_with_hash_bucket(
        'feature10', hash_bucket_size=2)
    list_of_inputs.append(copy.deepcopy({'categorical_column': cat_col_10}))

    # Using categorical_column_with_identity
    # Input 11: Basic identity column
    cat_col_11 = tf.feature_column.categorical_column_with_identity(
        'feature11', num_buckets=8)
    list_of_inputs.append(copy.deepcopy({'categorical_column': cat_col_11}))
    
    # Input 12: Identity column with a default value
    cat_col_12 = tf.feature_column.categorical_column_with_identity(
        'feature12', num_buckets=10, default_value=0)
    list_of_inputs.append(copy.deepcopy({'categorical_column': cat_col_12}))

    return list_of_inputs

generated_inputs["tf.feature_column.indicator_column"] = tf_feature_column_indicator_column_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.feature_column.indicator_column' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.indicator_column'.")

check_valid('tf.feature_column.indicator_column', generated_inputs['tf.feature_column.indicator_column'], lib="tf", suffix=0)
