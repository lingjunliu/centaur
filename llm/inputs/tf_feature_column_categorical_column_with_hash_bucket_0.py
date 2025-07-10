
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_categorical_column_with_hash_bucket_inputs():
    list_of_inputs = []

    # Input 1: Basic string type
    input_dict = {
        "key": "feature_1",
        "hash_bucket_size": 100,
        "dtype": tf.string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic int type
    input_dict = {
        "key": "feature_2",
        "hash_bucket_size": 200,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different key name
    input_dict = {
        "key": "user_id",
        "hash_bucket_size": 500,
        "dtype": tf.string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large hash_bucket_size
    input_dict = {
        "key": "product_id",
        "hash_bucket_size": 100000,
        "dtype": tf.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small hash_bucket_size
    input_dict = {
        "key": "category",
        "hash_bucket_size": 2,
        "dtype": tf.string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Another string example
    input_dict = {
        "key": "city",
        "hash_bucket_size": 50,
        "dtype": tf.string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Another int example
    input_dict = {
        "key": "age",
        "hash_bucket_size": 150,
        "dtype": tf.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: Different key name
    input_dict = {
        "key": "session_id",
        "hash_bucket_size": 75,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different hash bucket size
    input_dict = {
        "key": "country",
        "hash_bucket_size": 250,
        "dtype": tf.string
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different hash bucket size and dtype
    input_dict = {
        "key": "zip_code",
        "hash_bucket_size": 300,
        "dtype": tf.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.categorical_column_with_hash_bucket"] = tf_feature_column_categorical_column_with_hash_bucket_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.categorical_column_with_hash_bucket' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.categorical_column_with_hash_bucket'.")

check_valid('tf.feature_column.categorical_column_with_hash_bucket', generated_inputs['tf.feature_column.categorical_column_with_hash_bucket'], lib="tf", suffix=0)
