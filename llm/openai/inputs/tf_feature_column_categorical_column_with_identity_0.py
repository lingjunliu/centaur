
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_categorical_column_with_identity_inputs():
    list_of_inputs = []

    key = "user_id"
    num_buckets = np.int32(10)
    default_value = np.int32(0)
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "video_id"
    num_buckets = 1000000
    default_value = 0
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "session_index"
    num_buckets = np.int64(3)
    default_value = np.int64(2)
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "item"
    num_buckets = 255
    default_value = 1
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "country_code"
    num_buckets = 5
    default_value = 1
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "feature/segment"
    num_buckets = np.int16(2)
    default_value = np.int16(0)
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "edge_case_zero"
    num_buckets = 1
    default_value = 0
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "product_id"
    num_buckets = np.int64(1024)
    default_value = np.int64(123)
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "bucketed_age"
    num_buckets = np.int32(100)
    default_value = np.int32(99)
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "ad_slot"
    num_buckets = 7
    default_value = 3
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "city_hash"
    num_buckets = 2048
    default_value = 1024
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "experiment_group"
    num_buckets = np.int32(4)
    default_value = np.int32(0)
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    return list_of_inputs

generated_inputs["tf.feature_column.categorical_column_with_identity"] = tf_feature_column_categorical_column_with_identity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.feature_column.categorical_column_with_identity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.categorical_column_with_identity'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.feature_column.categorical_column_with_identity', generated_inputs['tf.feature_column.categorical_column_with_identity'], lib="tf", suffix=0)
