
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_categorical_column_with_identity_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    input_dict = {
        "key": "video_id",
        "num_buckets": 1000,
        "default_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different key
    input_dict = {
        "key": "user_id",
        "num_buckets": 500,
        "default_value": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different num_buckets and default_value
    input_dict = {
        "key": "item_id",
        "num_buckets": 10000,
        "default_value": 999
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small num_buckets
    input_dict = {
        "key": "category_id",
        "num_buckets": 10,
        "default_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger default value
    input_dict = {
        "key": "ad_id",
        "num_buckets": 2000,
        "default_value": 1999
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: default value in the middle
    input_dict = {
        "key": "feature_id",
        "num_buckets": 1500,
        "default_value": 750
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Another different key
    input_dict = {
        "key": "product_id",
        "num_buckets": 750,
        "default_value": 25
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Prime Number of buckets
    input_dict = {
        "key": "group_id",
        "num_buckets": 101,
        "default_value": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large number of buckets
    input_dict = {
        "key": "session_id",
        "num_buckets": 500000,
        "default_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another different key
    input_dict = {
        "key": "location_id",
        "num_buckets": 250,
        "default_value": 120
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.categorical_column_with_identity"] = tf_feature_column_categorical_column_with_identity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.categorical_column_with_identity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.categorical_column_with_identity'.")

check_valid('tf.feature_column.categorical_column_with_identity', generated_inputs['tf.feature_column.categorical_column_with_identity'], lib="tf", suffix=0)
