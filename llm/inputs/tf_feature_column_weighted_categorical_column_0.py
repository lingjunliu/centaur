
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_weighted_categorical_column_inputs():
    list_of_inputs = []

    # Helper function to create a dummy categorical column
    def create_dummy_categorical_column(column_name='terms', hash_bucket_size=1000):
        return tf.feature_column.categorical_column_with_hash_bucket(column_name, hash_bucket_size=hash_bucket_size)

    # Input 1, valid
    categorical_column = [create_dummy_categorical_column()]
    weight_feature_key = 'frequencies'
    dtype = tf.float32

    input_dict = {
        "categorical_column": categorical_column,
        "weight_feature_key": weight_feature_key,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid, different dtype
    categorical_column = [create_dummy_categorical_column()]
    weight_feature_key = 'weights'
    dtype = tf.float64

    input_dict = {
        "categorical_column": categorical_column,
        "weight_feature_key": weight_feature_key,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid, different column name
    categorical_column = [create_dummy_categorical_column(column_name='words')]
    weight_feature_key = 'freq'
    dtype = tf.float32

    input_dict = {
        "categorical_column": categorical_column,
        "weight_feature_key": weight_feature_key,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 4, valid, different hash bucket size
    categorical_column = [create_dummy_categorical_column(hash_bucket_size=500)]
    weight_feature_key = 'weights'
    dtype = tf.float32

    input_dict = {
        "categorical_column": categorical_column,
        "weight_feature_key": weight_feature_key,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, multiple categorical columns
    categorical_column = [create_dummy_categorical_column(), create_dummy_categorical_column(column_name='other_terms')]
    weight_feature_key = 'combined_weights'
    dtype = tf.float32

    input_dict = {
        "categorical_column": categorical_column,
        "weight_feature_key": weight_feature_key,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid, int32 dtype
    categorical_column = [create_dummy_categorical_column()]
    weight_feature_key = 'ints'
    dtype = tf.int32

    input_dict = {
        "categorical_column": categorical_column,
        "weight_feature_key": weight_feature_key,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7, valid, int64 dtype
    categorical_column = [create_dummy_categorical_column()]
    weight_feature_key = 'longs'
    dtype = tf.int64

    input_dict = {
        "categorical_column": categorical_column,
        "weight_feature_key": weight_feature_key,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8, valid, categorical column list with multiple items and diffrent column names
    categorical_column = [create_dummy_categorical_column(column_name = 'col1'), create_dummy_categorical_column(column_name = 'col2')]
    weight_feature_key = 'key1'
    dtype = tf.float32

    input_dict = {
        "categorical_column": categorical_column,
        "weight_feature_key": weight_feature_key,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid, Very small hash bucket size
    categorical_column = [create_dummy_categorical_column(hash_bucket_size=2)]
    weight_feature_key = 'small_weights'
    dtype = tf.float32

    input_dict = {
        "categorical_column": categorical_column,
        "weight_feature_key": weight_feature_key,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid, large hash bucket size
    categorical_column = [create_dummy_categorical_column(hash_bucket_size=100000)]
    weight_feature_key = 'large_weights'
    dtype = tf.float32

    input_dict = {
        "categorical_column": categorical_column,
        "weight_feature_key": weight_feature_key,
        "dtype": dtype
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
