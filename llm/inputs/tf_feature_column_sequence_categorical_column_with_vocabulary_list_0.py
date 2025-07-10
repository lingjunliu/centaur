
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_sequence_categorical_column_with_vocabulary_list_inputs():
    list_of_inputs = []

    # Input 1
    key = 'color'
    vocabulary_list = ['red', 'green', 'blue']
    dtype = np.dtype(np.object_)
    default_value = -1
    num_oov_buckets = 0
    input_dict = {'key': key, 'vocabulary_list': vocabulary_list, 'dtype': dtype, 'default_value': default_value, 'num_oov_buckets': num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    key = 'number'
    vocabulary_list = [1, 2, 3, 4]
    dtype = np.dtype(np.int64)
    default_value = 0
    num_oov_buckets = 0
    input_dict = {'key': key, 'vocabulary_list': vocabulary_list, 'dtype': dtype, 'default_value': default_value, 'num_oov_buckets': num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    key = 'city'
    vocabulary_list = ['london', 'paris', 'tokyo']
    dtype = np.dtype(np.object_)
    default_value = -1
    num_oov_buckets = 2
    input_dict = {'key': key, 'vocabulary_list': vocabulary_list, 'dtype': dtype, 'default_value': -1, 'num_oov_buckets': num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    key = 'age'
    vocabulary_list = [20, 25, 30, 35]
    dtype = np.dtype(np.int32)
    default_value = -1
    num_oov_buckets = 1
    input_dict = {'key': key, 'vocabulary_list': vocabulary_list, 'dtype': dtype, 'default_value': -1, 'num_oov_buckets': num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    key = 'product'
    vocabulary_list = ['A', 'B', 'C']
    dtype = np.dtype(np.object_)
    default_value = -1
    num_oov_buckets = 0
    input_dict = {'key': key, 'vocabulary_list': vocabulary_list, 'dtype': dtype, 'default_value': default_value, 'num_oov_buckets': num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    key = 'size'
    vocabulary_list = [10, 20, 30]
    dtype = np.dtype(np.int64)
    default_value = -1
    num_oov_buckets = 0
    input_dict = {'key': key, 'vocabulary_list': vocabulary_list, 'dtype': dtype, 'default_value': default_value, 'num_oov_buckets': num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    key = 'day'
    vocabulary_list = ['mon', 'tue', 'wed', 'thu', 'fri']
    dtype = np.dtype(np.object_)
    default_value = -1
    num_oov_buckets = 1
    input_dict = {'key': key, 'vocabulary_list': vocabulary_list, 'dtype': dtype, 'default_value': default_value, 'num_oov_buckets': num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    key = 'count'
    vocabulary_list = [1, 5, 10, 20]
    dtype = np.dtype(np.int32)
    default_value = 0
    num_oov_buckets = 0
    input_dict = {'key': key, 'vocabulary_list': vocabulary_list, 'dtype': dtype, 'default_value': default_value, 'num_oov_buckets': num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    key = 'state'
    vocabulary_list = ['CA', 'NY', 'TX']
    dtype = np.dtype(np.object_)
    default_value = -1
    num_oov_buckets = 3
    input_dict = {'key': key, 'vocabulary_list': vocabulary_list, 'dtype': dtype, 'default_value': -1, 'num_oov_buckets': num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    key = 'score'
    vocabulary_list = [100, 200, 300, 400]
    dtype = np.dtype(np.int64)
    default_value = 1
    num_oov_buckets = 0
    input_dict = {'key': key, 'vocabulary_list': vocabulary_list, 'dtype': dtype, 'default_value': default_value, 'num_oov_buckets': num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.sequence_categorical_column_with_vocabulary_list"] = tf_feature_column_sequence_categorical_column_with_vocabulary_list_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.sequence_categorical_column_with_vocabulary_list' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.sequence_categorical_column_with_vocabulary_list'.")

check_valid('tf.feature_column.sequence_categorical_column_with_vocabulary_list', generated_inputs['tf.feature_column.sequence_categorical_column_with_vocabulary_list'], lib="tf", suffix=0)
