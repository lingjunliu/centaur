
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_weighted_categorical_column_inputs():
    list_of_inputs = []

    # Input 1, valid
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(
        key='terms1', hash_bucket_size=100)]
    weight_feature_key = 'frequencies1'
    dtype = np.float32

    input_dict = {
        "categorical_column": categorical_column,
        "weight_feature_key": weight_feature_key,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_list(
        key='terms2', vocabulary_list=['a', 'b', 'c'])]
    weight_feature_key = 'frequencies2'
    dtype = np.float64

    input_dict = {
        "categorical_column": categorical_column,
        "weight_feature_key": weight_feature_key,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    categorical_column = [tf.feature_column.categorical_column_with_identity(
        key='terms3', num_buckets=5)]
    weight_feature_key = 'frequencies3'
    dtype = np.float16

    input_dict = {
        "categorical_column": categorical_column,
        "weight_feature_key": weight_feature_key,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(
        key='terms4', hash_bucket_size=2000)]
    weight_feature_key = 'frequencies4'
    dtype = np.int32

    input_dict = {
        "categorical_column": categorical_column,
        "weight_feature_key": weight_feature_key,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_file(
        key='terms5', vocabulary_file='my_vocab_file.txt', vocabulary_size=10)]
    weight_feature_key = 'frequencies5'
    dtype = np.int64

    input_dict = {
        "categorical_column": categorical_column,
        "weight_feature_key": weight_feature_key,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(
        key='terms6', hash_bucket_size=500)]
    weight_feature_key = 'frequencies6'
    dtype = np.float32

    input_dict = {
        "categorical_column": categorical_column,
        "weight_feature_key": weight_feature_key,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_list(
        key='terms7', vocabulary_list=['x', 'y', 'z', 'w'])]
    weight_feature_key = 'frequencies7'
    dtype = np.float64

    input_dict = {
        "categorical_column": categorical_column,
        "weight_feature_key": weight_feature_key,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8, valid, multiple categorical columns
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(
        key='terms8_1', hash_bucket_size=100),
        tf.feature_column.categorical_column_with_hash_bucket(
        key='terms8_2', hash_bucket_size=200)]
    weight_feature_key = 'frequencies8'
    dtype = np.float32

    input_dict = {
        "categorical_column": categorical_column,
        "weight_feature_key": weight_feature_key,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid, multiple categorical columns
    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_list(
        key='terms9_1', vocabulary_list=['a', 'b', 'c']),
        tf.feature_column.categorical_column_with_identity(
        key='terms9_2', num_buckets=10)]
    weight_feature_key = 'frequencies9'
    dtype = np.float64

    input_dict = {
        "categorical_column": categorical_column,
        "weight_feature_key": weight_feature_key,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid, numpy dtype
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(
        key='terms10', hash_bucket_size=300)]
    weight_feature_key = 'frequencies10'
    dtype = np.int32

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
