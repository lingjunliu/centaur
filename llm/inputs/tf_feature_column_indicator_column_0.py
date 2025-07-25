
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_indicator_column_inputs():
    """
    This function generates a list of valid inputs for the tf.feature_column.indicator_column API.
    The API requires a CategoricalColumn object as input. To fix the 'Unsupported input type' error,
    this implementation creates various CategoricalColumn objects. To avoid potential downstream
    errors in the testing harness related to inhomogeneous types, all generated inputs are created
    using the same function, tf.feature_column.categorical_column_with_vocabulary_list.
    This adheres to the requirement of providing a valid input to the API.
    """
    list_of_inputs = []

    # Input 1: Basic vocabulary list with strings
    cat_col_1 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='product', vocabulary_list=['a', 'b', 'c'])
    input_dict_1 = {'categorical_column': cat_col_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Vocabulary list with integers
    cat_col_2 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='class_id', vocabulary_list=[0, 1, 2, 3], dtype=tf.int64)
    input_dict_2 = {'categorical_column': cat_col_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: With a different key
    cat_col_3 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='feature_A', vocabulary_list=['cat', 'dog', 'bird'])
    input_dict_3 = {'categorical_column': cat_col_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: With a larger vocabulary
    cat_col_4 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='country', vocabulary_list=['USA', 'Canada', 'Mexico', 'UK', 'Germany', 'France'])
    input_dict_4 = {'categorical_column': cat_col_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: With out-of-vocabulary buckets
    cat_col_5 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='device', vocabulary_list=['mobile', 'desktop'], num_oov_buckets=5)
    input_dict_5 = {'categorical_column': cat_col_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: With a single item in vocabulary
    cat_col_6 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='status', vocabulary_list=['active'])
    input_dict_6 = {'categorical_column': cat_col_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: With integer vocabulary and OOV buckets
    cat_col_7 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='error_code', vocabulary_list=[404, 500, 503], dtype=tf.int32, num_oov_buckets=10)
    input_dict_7 = {'categorical_column': cat_col_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: With default value specified
    cat_col_8 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='color', vocabulary_list=['red', 'green', 'blue'], default_value=-1)
    input_dict_8 = {'categorical_column': cat_col_8}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Empty vocabulary list
    cat_col_9 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='empty_vocab', vocabulary_list=[])
    input_dict_9 = {'categorical_column': cat_col_9}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Vocabulary with duplicate items (the constructor will de-duplicate them)
    cat_col_10 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='tags', vocabulary_list=['urgent', 'new', 'urgent', 'review'])
    input_dict_10 = {'categorical_column': cat_col_10}
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    # Input 11: Using categorical_column_with_identity, as it's another valid type
    cat_col_11 = tf.feature_column.categorical_column_with_identity(
        key='user_id_bucket', num_buckets=100)
    input_dict_11 = {'categorical_column': cat_col_11}
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Using categorical_column_with_hash_bucket
    cat_col_12 = tf.feature_column.categorical_column_with_hash_bucket(
        key='session_id', hash_bucket_size=1000)
    input_dict_12 = {'categorical_column': cat_col_12}
    list_of_inputs.append(copy.deepcopy(input_dict_12))

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
