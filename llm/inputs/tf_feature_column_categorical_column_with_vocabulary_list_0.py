
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np

def tf_feature_column_categorical_column_with_vocabulary_list_inputs():
    list_of_inputs = []

    input_dict = {
        'key': 'color',
        'vocabulary_list': ['red', 'green', 'blue'],
        'dtype': np.dtype(np.str_),
        'default_value': -1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(input_dict)

    input_dict = {
        'key': 'number',
        'vocabulary_list': [1, 2, 3, 4],
        'dtype': np.dtype(np.int64),
        'default_value': 0,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(input_dict)

    input_dict = {
        'key': 'shape',
        'vocabulary_list': ['square', 'circle', 'triangle'],
        'dtype': np.dtype(np.str_),
        'default_value': -1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(input_dict)

    input_dict = {
        'key': 'fruit',
        'vocabulary_list': ['apple', 'banana', 'orange'],
        'dtype': np.dtype(np.str_),
        'default_value': 0,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(input_dict)

    input_dict = {
        'key': 'size',
        'vocabulary_list': [10, 20, 30, 40, 50],
        'dtype': np.dtype(np.int64),
        'default_value': -1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(input_dict)

    input_dict = {
        'key': 'animal',
        'vocabulary_list': ['dog', 'cat', 'bird', 'fish'],
        'dtype': np.dtype(np.str_),
        'default_value': -1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(input_dict)

    input_dict = {
        'key': 'city',
        'vocabulary_list': ['london', 'paris', 'tokyo'],
        'dtype': np.dtype(np.str_),
        'default_value': 1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(input_dict)

    input_dict = {
        'key': 'score',
        'vocabulary_list': [100, 200, 300, 400],
        'dtype': np.dtype(np.int64),
        'default_value': -1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(input_dict)

    input_dict = {
        'key': 'day',
        'vocabulary_list': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'],
        'dtype': np.dtype(np.str_),
        'default_value': -1,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(input_dict)

    input_dict = {
        'key': 'count',
        'vocabulary_list': [1, 5, 10, 15, 20, 25],
        'dtype': np.dtype(np.int64),
        'default_value': 0,
        'num_oov_buckets': 0
    }
    list_of_inputs.append(input_dict)

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
