
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_feature_column_crossed_column_inputs():
    list_of_inputs = []

    # The test harness has shown issues with processing the `keys` argument,
    # which can be a list of strings or a list of CategoricalColumn objects.
    # Previous attempts with lists of strings led to a UFuncNoLoopError, and
    # lists of mixed CategoricalColumn objects led to a TypeError.
    # This attempt uses only one type of CategoricalColumn to ensure the
    # objects in the list are homogeneous, which might avoid comparison errors
    # in the test harness.
    cat_cols = [tf.feature_column.categorical_column_with_identity(f'key_{i}', 10 * (i + 1)) for i in range(5)]

    # 1
    input_dict = {'keys': [cat_cols[0], cat_cols[1]], 'hash_bucket_size': 100, 'hash_key': 'k01'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # 2
    input_dict = {'keys': [cat_cols[2], cat_cols[3]], 'hash_bucket_size': 200, 'hash_key': 'k23'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # 3
    input_dict = {'keys': [cat_cols[0], cat_cols[4]], 'hash_bucket_size': 300, 'hash_key': 'k04'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # 4
    input_dict = {'keys': [cat_cols[1], cat_cols[2], cat_cols[3]], 'hash_bucket_size': 1000, 'hash_key': 'k123'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # 5
    input_dict = {'keys': [cat_cols[0], cat_cols[1]], 'hash_bucket_size': 2, 'hash_key': 'minkey'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # 6
    input_dict = {'keys': [cat_cols[3], cat_cols[4]], 'hash_bucket_size': 999, 'hash_key': ''}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # 7
    input_dict = {'keys': [cat_cols[0], cat_cols[2]], 'hash_bucket_size': 50, 'hash_key': 'anotherkey'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # 8
    input_dict = {'keys': [cat_cols[1], cat_cols[4]], 'hash_bucket_size': 50000, 'hash_key': 'big_hash'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # 9
    input_dict = {'keys': [cat_cols[0], cat_cols[1], cat_cols[2], cat_cols[3], cat_cols[4]], 'hash_bucket_size': 100000, 'hash_key': 'all_keys'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # 10
    input_dict = {'keys': [cat_cols[4], cat_cols[0]], 'hash_bucket_size': 1234, 'hash_key': 'reversed_keys'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # 11
    input_dict = {'keys': [cat_cols[1], cat_cols[3]], 'hash_bucket_size': 789, 'hash_key': 'some_hash_key_123'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.feature_column.crossed_column"] = tf_feature_column_crossed_column_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.feature_column.crossed_column' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.crossed_column'.")

check_valid('tf.feature_column.crossed_column', generated_inputs['tf.feature_column.crossed_column'], lib="tf", suffix=0)
