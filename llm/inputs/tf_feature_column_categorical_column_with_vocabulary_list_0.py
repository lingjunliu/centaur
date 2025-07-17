
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_categorical_column_with_vocabulary_list_inputs():
    list_of_inputs = []

    # Input 1: string vocabulary, default default_value
    input_dict = {
        "key": "color",
        "vocabulary_list": ["red", "green", "blue"],
        "dtype": tf.string,
        "default_value": -1,
        "num_oov_buckets": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int vocabulary, no default_value, num_oov_buckets
    input_dict = {
        "key": "number",
        "vocabulary_list": [1, 2, 3, 4],
        "dtype": tf.int64,
        "default_value": -1,
        "num_oov_buckets": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: string vocabulary, num_oov_buckets > 0
    input_dict = {
        "key": "city",
        "vocabulary_list": ["london", "paris", "tokyo"],
        "dtype": tf.string,
        "default_value": -1,
        "num_oov_buckets": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int vocabulary, negative numbers, num_oov_buckets
    input_dict = {
        "key": "temperature",
        "vocabulary_list": [-10, -5, 0, 5, 10],
        "dtype": tf.int64,
        "default_value": -1,
        "num_oov_buckets": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: string vocabulary, longer vocabulary,  num_oov_buckets
    input_dict = {
        "key": "animal",
        "vocabulary_list": ["dog", "cat", "bird", "fish", "hamster", "gerbil"],
        "dtype": tf.string,
        "default_value": -1,
        "num_oov_buckets": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: int vocabulary, num_oov_buckets larger number
    input_dict = {
        "key": "id",
        "vocabulary_list": [100, 200, 300],
        "dtype": tf.int64,
        "default_value": -1,
        "num_oov_buckets": 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: string vocabulary, no empty string
    input_dict = {
        "key": "text",
        "vocabulary_list": ["hello", "world"],
        "dtype": tf.string,
        "default_value": -1,
        "num_oov_buckets": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int vocabulary, default_value 1000
    input_dict = {
        "key": "user_id",
        "vocabulary_list": [1, 2, 3],
        "dtype": tf.int64,
        "default_value": 1000,
        "num_oov_buckets": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: string vocabulary, unicode chars,  num_oov_buckets
    input_dict = {
        "key": "unicode_test",
        "vocabulary_list": ["你好", "こんにちは", "안녕하세요"],
        "dtype": tf.string,
        "default_value": -1,
        "num_oov_buckets": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int vocabulary, small numbers, num_oov_buckets
    input_dict = {
        "key": "small_number",
        "vocabulary_list": [0, 1, 2],
        "dtype": tf.int64,
        "default_value": -1,
        "num_oov_buckets": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.categorical_column_with_vocabulary_list"] = tf_feature_column_categorical_column_with_vocabulary_list_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.categorical_column_with_vocabulary_list' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.categorical_column_with_vocabulary_list'.")

check_valid('tf.feature_column.categorical_column_with_vocabulary_list', generated_inputs['tf.feature_column.categorical_column_with_vocabulary_list'], lib="tf", suffix=0)
