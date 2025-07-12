
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_weighted_categorical_column_inputs():
    list_of_inputs = []

    # Input 1
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(
        key='terms', hash_bucket_size=1000)]
    weight_feature_key = 'frequencies'
    dtype = np.float32

    input_dict = {
        'categorical_column': categorical_column,
        'weight_feature_key': weight_feature_key,
        'dtype': dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    categorical_column = [tf.feature_column.categorical_column_with_identity(
        key='user_id', num_buckets=100)]
    weight_feature_key = 'weight'
    dtype = np.float64

    input_dict = {
        'categorical_column': categorical_column,
        'weight_feature_key': weight_feature_key,
        'dtype': dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_list(
        key='city', vocabulary_list=['NYC', 'SF', 'LA'])]
    weight_feature_key = 'importance'
    dtype = np.int32

    input_dict = {
        'categorical_column': categorical_column,
        'weight_feature_key': weight_feature_key,
        'dtype': dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(
        key='product_id', hash_bucket_size=5000)]
    weight_feature_key = 'quantity'
    dtype = np.float16

    input_dict = {
        'categorical_column': categorical_column,
        'weight_feature_key': weight_feature_key,
        'dtype': dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_file(
        key='words', vocabulary_file='vocab.txt', vocabulary_size=10000)]
    weight_feature_key = 'relevance'
    dtype = np.int64

    input_dict = {
        'categorical_column': categorical_column,
        'weight_feature_key': weight_feature_key,
        'dtype': dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(
        key='feature1', hash_bucket_size=200)]
    weight_feature_key = 'feature2'
    dtype = np.float32
    input_dict = {
        'categorical_column': categorical_column,
        'weight_feature_key': weight_feature_key,
        'dtype': dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    categorical_column = [tf.feature_column.categorical_column_with_identity(
        key='id', num_buckets=50)]
    weight_feature_key = 'value'
    dtype = np.int16
    input_dict = {
        'categorical_column': categorical_column,
        'weight_feature_key': weight_feature_key,
        'dtype': dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_list(
        key='color', vocabulary_list=['red', 'green', 'blue'])]
    weight_feature_key = 'intensity'
    dtype = np.float32
    input_dict = {
        'categorical_column': categorical_column,
        'weight_feature_key': weight_feature_key,
        'dtype': dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(key="test_feature", hash_bucket_size=1024)]
    weight_feature_key = "importance_score"
    dtype = np.float32
    input_dict = {
        'categorical_column': categorical_column,
        'weight_feature_key': weight_feature_key,
        'dtype': dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    categorical_column = [tf.feature_column.categorical_column_with_identity(key="category_id", num_buckets=256)]
    weight_feature_key = "relative_weight"
    dtype = np.float64
    input_dict = {
        'categorical_column': categorical_column,
        'weight_feature_key': weight_feature_key,
        'dtype': dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.weighted_categorical_column"] = tf_feature_column_weighted_categorical_column_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.weighted_categorical_column' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.weighted_categorical_column'.")

check_valid('tf.feature_column.weighted_categorical_column', generated_inputs['tf.feature_column.weighted_categorical_column'], lib="tf", suffix=0)
