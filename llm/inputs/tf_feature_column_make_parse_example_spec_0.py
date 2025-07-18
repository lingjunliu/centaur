
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np
import os

def tf_feature_column_make_parse_example_spec_inputs():
    """
    Generates a list of valid inputs for the
    tf.feature_column.make_parse_example_spec function.
    """
    list_of_inputs = []

    # Input 1: Single numeric column with default parameters
    fc1 = tf.compat.v1.feature_column.numeric_column(key="price")
    input_dict_1 = {'feature_columns': [fc1]}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: A numeric column with a non-default shape
    fc2 = tf.compat.v1.feature_column.numeric_column(key="image", shape=(28, 28))
    input_dict_2 = {'feature_columns': [fc2]}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Categorical column with an inline vocabulary list
    vocab_list_3 = ['cat', 'dog', 'mouse']
    fc3 = tf.compat.v1.feature_column.categorical_column_with_vocabulary_list(
        key="animal", vocabulary_list=vocab_list_3)
    input_dict_3 = {'feature_columns': [fc3]}
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    # Input 4: Categorical column with identity for integer categories
    fc4 = tf.compat.v1.feature_column.categorical_column_with_identity(
        key="user_id", num_buckets=1000)
    input_dict_4 = {'feature_columns': [fc4]}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: A bucketized column.
    source_fc5 = tf.compat.v1.feature_column.numeric_column("feature_c")
    fc5 = tf.compat.v1.feature_column.bucketized_column(
        source_column=source_fc5, boundaries=[0.0, 10.0, 20.0])
    input_dict_5 = {'feature_columns': [fc5]}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: A crossed column.
    source_fc6a = tf.compat.v1.feature_column.categorical_column_with_vocabulary_list(
        key="feature_a", vocabulary_list=['a', 'b'])
    source_fc6b_numeric = tf.compat.v1.feature_column.numeric_column("feature_b")
    source_fc6b_bucketized = tf.compat.v1.feature_column.bucketized_column(
        source_column=source_fc6b_numeric, boundaries=[10, 20])
    fc6 = tf.compat.v1.feature_column.crossed_column(
        keys=[source_fc6a, source_fc6b_bucketized], hash_bucket_size=100)
    input_dict_6 = {'feature_columns': [fc6]}
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: An indicator column.
    source_fc7 = tf.compat.v1.feature_column.categorical_column_with_vocabulary_list(
        key="color", vocabulary_list=["red", "green", "blue"])
    fc7 = tf.compat.v1.feature_column.indicator_column(source_fc7)
    input_dict_7 = {'feature_columns': [fc7]}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: An embedding column.
    source_fc8 = tf.compat.v1.feature_column.categorical_column_with_hash_bucket(
        key="product_id", hash_bucket_size=50)
    fc8 = tf.compat.v1.feature_column.embedding_column(source_fc8, dimension=8)
    input_dict_8 = {'feature_columns': [fc8]}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: An empty list of feature columns (edge case)
    input_dict_9 = {'feature_columns': []}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Numeric column with a different dtype (int64)
    fc10 = tf.compat.v1.feature_column.numeric_column(key="item_count", dtype=tf.int64)
    input_dict_10 = {'feature_columns': [fc10]}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Weighted categorical column
    ids = tf.compat.v1.feature_column.categorical_column_with_identity('video_id', num_buckets=100)
    weighted_col = tf.compat.v1.feature_column.weighted_categorical_column(ids, 'video_watch_time')
    input_dict_11 = {'feature_columns': [weighted_col]}
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Categorical column with vocabulary file
    dummy_vocab_file = '/tmp/dummy_vocab.txt'
    with open(dummy_vocab_file, 'w') as f:
        f.write("product_a\n")
        f.write("product_b\n")
    vocab_col = tf.compat.v1.feature_column.categorical_column_with_vocabulary_file(
        key='product_name', vocabulary_file=dummy_vocab_file, num_oov_buckets=1)
    input_dict_12 = {'feature_columns': [vocab_col]}
    list_of_inputs.append(copy.deepcopy(input_dict_12))
    os.remove(dummy_vocab_file)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.make_parse_example_spec"] = tf_feature_column_make_parse_example_spec_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.feature_column.make_parse_example_spec' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.make_parse_example_spec'.")

check_valid('tf.feature_column.make_parse_example_spec', generated_inputs['tf.feature_column.make_parse_example_spec'], lib="tf", suffix=0)
