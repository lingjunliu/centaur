
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_sequence_categorical_column_with_hash_bucket_inputs():
    list_of_inputs = []

    input_dict = {
        "key": "tokens",
        "hash_bucket_size": 1000,
        "dtype": np.dtype("U10")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "words",
        "hash_bucket_size": np.int32(2),
        "dtype": np.dtype("S8")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "ids",
        "hash_bucket_size": 17,
        "dtype": np.dtype("int32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "categories_en",
        "hash_bucket_size": np.int64(4096),
        "dtype": np.dtype("int64")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "序列",
        "hash_bucket_size": 257,
        "dtype": np.dtype(np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "byte_tokens",
        "hash_bucket_size": 65535,
        "dtype": np.dtype("S1")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "click_ids",
        "hash_bucket_size": 100,
        "dtype": np.dtype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "product_ids",
        "hash_bucket_size": np.int32(8192),
        "dtype": np.dtype(np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "tags",
        "hash_bucket_size": 3,
        "dtype": np.dtype("U4")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "features",
        "hash_bucket_size": 50,
        "dtype": np.dtype("U1")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "labels",
        "hash_bucket_size": 1024,
        "dtype": np.dtype("S16")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "session_tokens",
        "hash_bucket_size": 200,
        "dtype": np.dtype("U32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.feature_column.sequence_categorical_column_with_hash_bucket"] = tf_feature_column_sequence_categorical_column_with_hash_bucket_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.feature_column.sequence_categorical_column_with_hash_bucket' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.sequence_categorical_column_with_hash_bucket'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.feature_column.sequence_categorical_column_with_hash_bucket', generated_inputs['tf.feature_column.sequence_categorical_column_with_hash_bucket'], lib="tf", suffix=0)
