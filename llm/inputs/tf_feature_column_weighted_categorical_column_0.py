
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
        key='terms1', hash_bucket_size=100)]
    weight_feature_key = 'frequencies1'
    dtype = tf.float32
    input_dict = {'categorical_column': categorical_column, 'weight_feature_key': weight_feature_key, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(
        key='terms2', hash_bucket_size=500)]
    weight_feature_key = 'frequencies2'
    dtype = tf.float64
    input_dict = {'categorical_column': categorical_column, 'weight_feature_key': weight_feature_key, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(
        key='terms3', hash_bucket_size=2000)]
    weight_feature_key = 'frequencies3'
    dtype = tf.int32
    input_dict = {'categorical_column': categorical_column, 'weight_feature_key': weight_feature_key, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_list(
        key='terms4', vocabulary_list=['a', 'b', 'c'])]
    weight_feature_key = 'frequencies4'
    dtype = tf.float32
    input_dict = {'categorical_column': categorical_column, 'weight_feature_key': weight_feature_key, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    categorical_column = [tf.feature_column.categorical_column_with_identity(
        key='terms5', num_buckets=10)]
    weight_feature_key = 'frequencies5'
    dtype = tf.float64
    input_dict = {'categorical_column': categorical_column, 'weight_feature_key': weight_feature_key, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(
        key='terms6', hash_bucket_size=10000)]
    weight_feature_key = 'frequencies6'
    dtype = tf.int32 # was int64, changed to int32 as it is a supported dtype
    input_dict = {'categorical_column': categorical_column, 'weight_feature_key': weight_feature_key, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Removed vocabulary file due to complexity and external dependency
    # Input 8
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(
        key='terms8', hash_bucket_size=50)]
    weight_feature_key = 'frequencies8'
    dtype = tf.float32  # Changed to supported dtype
    input_dict = {'categorical_column': categorical_column, 'weight_feature_key': weight_feature_key, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(
        key='terms9', hash_bucket_size=100000)]
    weight_feature_key = 'frequencies9'
    dtype = tf.int32 # Changed to supported dtype
    input_dict = {'categorical_column': categorical_column, 'weight_feature_key': weight_feature_key, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    categorical_column = [tf.feature_column.categorical_column_with_identity(
        key='terms10', num_buckets=100)]
    weight_feature_key = 'frequencies10'
    dtype = tf.int32 # Changed to supported dtype
    input_dict = {'categorical_column': categorical_column, 'weight_feature_key': weight_feature_key, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11 - Categorical column with vocabulary list, int dtype
    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_list(
        key='terms11', vocabulary_list=[1, 2, 3])]
    weight_feature_key = 'frequencies11'
    dtype = tf.int32
    input_dict = {'categorical_column': categorical_column, 'weight_feature_key': weight_feature_key, 'dtype': dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12 - Categorical column with identity, int dtype, smaller num_buckets
    categorical_column = [tf.feature_column.categorical_column_with_identity(
        key='terms12', num_buckets=5)]
    weight_feature_key = 'frequencies12'
    dtype = tf.int32
    input_dict = {'categorical_column': categorical_column, 'weight_feature_key': weight_feature_key, 'dtype': dtype}
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
