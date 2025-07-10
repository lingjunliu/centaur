
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_sequence_categorical_column_with_identity_inputs():
    list_of_inputs = []

    # Input 1, valid
    input_dict = {
        "key": "watches",
        "num_buckets": 1000,
        "default_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    input_dict = {
        "key": "products",
        "num_buckets": 500,
        "default_value": 499
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    input_dict = {
        "key": "cities",
        "num_buckets": 10,
        "default_value": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    input_dict = {
        "key": "countries",
        "num_buckets": 200,
        "default_value": 100
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    input_dict = {
        "key": "items",
        "num_buckets": 2,
        "default_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    input_dict = {
        "key": "categories",
        "num_buckets": 10000,
        "default_value": 9999
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7, valid
    input_dict = {
        "key": "age",
        "num_buckets": 100,
        "default_value": 50
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    input_dict = {
        "key": "gender",
        "num_buckets": 3,
        "default_value": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    input_dict = {
        "key": "occupation",
        "num_buckets": 50,
        "default_value": 25
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    input_dict = {
        "key": "ids",
        "num_buckets": 100000,
        "default_value": 50000
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.sequence_categorical_column_with_identity"] = tf_feature_column_sequence_categorical_column_with_identity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.sequence_categorical_column_with_identity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.sequence_categorical_column_with_identity'.")

check_valid('tf.feature_column.sequence_categorical_column_with_identity', generated_inputs['tf.feature_column.sequence_categorical_column_with_identity'], lib="tf", suffix=0)
