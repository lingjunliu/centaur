
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_categorical_column_with_vocabulary_list_inputs():
    list_of_inputs = []

    # Input 1: String vocabulary, default default_value
    input_dict = {
        "key": "color",
        "vocabulary_list": list(["red", "green", "blue"]),
        "dtype": np.dtype('U5'),
        "default_value": -1,
        "num_oov_buckets": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer vocabulary, default default_value
    input_dict = {
        "key": "number",
        "vocabulary_list": list([1, 2, 3, 4]),
        "dtype": np.dtype(np.int64),
        "default_value": 0,
        "num_oov_buckets": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: String vocabulary, num_oov_buckets
    input_dict = {
        "key": "city",
        "vocabulary_list": list(["london", "paris", "tokyo"]),
        "dtype": np.dtype('U6'),
        "default_value": -1,
        "num_oov_buckets": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Integer vocabulary, num_oov_buckets
    input_dict = {
        "key": "age",
        "vocabulary_list": list([20, 30, 40, 50]),
        "dtype": np.dtype(np.int32),
        "default_value": -1,
        "num_oov_buckets": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Mixed string vocabulary with special chars, default default_value
    input_dict = {
        "key": "item",
        "vocabulary_list": list(["item_1", "item-2", "item.3"]),
        "dtype": np.dtype('U6'),
        "default_value": 10,
        "num_oov_buckets": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6:  String vocabulary, with empty string, default default_value
    input_dict = {
        "key": "name",
        "vocabulary_list": list(["john", "jane", ""]),
        "dtype": np.dtype('U4'),
        "default_value": -1,
        "num_oov_buckets": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Longer vocabulary
    input_dict = {
        "key": "weekday",
        "vocabulary_list": list(["mon", "tue", "wed", "thu", "fri", "sat", "sun"]),
        "dtype": np.dtype('U3'),
        "default_value": 0,
        "num_oov_buckets": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Integer vocabulary with negative values
    input_dict = {
        "key": "temperature",
        "vocabulary_list": list([-10, 0, 10, 20, 30]),
        "dtype": np.dtype(np.int32),
        "default_value": -2,
        "num_oov_buckets": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: String vocabulary with uppercase and lowercase
    input_dict = {
        "key": "letter",
        "vocabulary_list": list(["a", "B", "c", "D"]),
        "dtype": np.dtype('U1'),
        "default_value": -1,
        "num_oov_buckets": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Integer, larger oov buckets
    input_dict = {
        "key": "id",
        "vocabulary_list": list([100, 200, 300]),
        "dtype": np.dtype(np.int64),
        "default_value": -1,
        "num_oov_buckets": 5
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
