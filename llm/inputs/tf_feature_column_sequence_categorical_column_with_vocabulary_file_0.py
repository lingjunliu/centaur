
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os

def tf_feature_column_sequence_categorical_column_with_vocabulary_file_inputs():
    list_of_inputs = []

    # Create dummy vocabulary file
    vocab_file = "vocab.txt"
    with open(vocab_file, "w") as f:
        f.write("apple\n")
        f.write("banana\n")
        f.write("cherry\n")
        f.write("date\n")
        f.write("elderberry\n")

    # Input 1
    key = "fruits"
    vocabulary_file = vocab_file
    vocabulary_size = 5
    num_oov_buckets = 0
    default_value = -1
    dtype = tf.string

    input_dict = {
        "key": key,
        "vocabulary_file": vocabulary_file,
        "vocabulary_size": np.int32(vocabulary_size),
        "num_oov_buckets": np.int32(num_oov_buckets),
        "default_value": np.int32(default_value),
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    key = "cities"
    vocabulary_file = vocab_file
    vocabulary_size = 3
    num_oov_buckets = 2
    default_value = None
    dtype = tf.string

    input_dict = {
        "key": key,
        "vocabulary_file": vocabulary_file,
        "vocabulary_size": np.int32(vocabulary_size),
        "num_oov_buckets": np.int32(num_oov_buckets),
        "default_value": default_value,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    key = "numbers"
    vocabulary_file = vocab_file
    vocabulary_size = 1
    num_oov_buckets = 0
    default_value = 0
    dtype = tf.string
    
    input_dict = {
        "key": key,
        "vocabulary_file": vocabulary_file,
        "vocabulary_size": np.int32(vocabulary_size),
        "num_oov_buckets": np.int32(num_oov_buckets),
        "default_value": np.int32(default_value),
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    key = "states"
    vocabulary_file = vocab_file
    vocabulary_size = 5
    num_oov_buckets = 0
    default_value = -1
    dtype = tf.string
    
    input_dict = {
        "key": key,
        "vocabulary_file": vocabulary_file,
        "vocabulary_size": np.int32(vocabulary_size),
        "num_oov_buckets": np.int32(num_oov_buckets),
        "default_value": np.int32(default_value),
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    # Input 5
    key = "symbols"
    vocabulary_file = vocab_file
    vocabulary_size = 2
    num_oov_buckets = 3
    default_value = None
    dtype = tf.string

    input_dict = {
        "key": key,
        "vocabulary_file": vocabulary_file,
        "vocabulary_size": np.int32(vocabulary_size),
        "num_oov_buckets": np.int32(num_oov_buckets),
        "default_value": default_value,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (Integer dtype)
    key = "ids"
    vocabulary_file = vocab_file
    vocabulary_size = 5
    num_oov_buckets = 0
    default_value = -1
    dtype = tf.int64
    
    input_dict = {
        "key": key,
        "vocabulary_file": vocabulary_file,
        "vocabulary_size": np.int32(vocabulary_size),
        "num_oov_buckets": np.int32(num_oov_buckets),
        "default_value": np.int32(default_value),
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (Integer dtype with oov)
    key = "item_ids"
    vocabulary_file = vocab_file
    vocabulary_size = 3
    num_oov_buckets = 2
    default_value = None
    dtype = tf.int64

    input_dict = {
        "key": key,
        "vocabulary_file": vocabulary_file,
        "vocabulary_size": np.int32(vocabulary_size),
        "num_oov_buckets": np.int32(num_oov_buckets),
        "default_value": default_value,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (Integer dtype with default_value)
    key = "user_ids"
    vocabulary_file = vocab_file
    vocabulary_size = 4
    num_oov_buckets = 0
    default_value = 0
    dtype = tf.int64
    
    input_dict = {
        "key": key,
        "vocabulary_file": vocabulary_file,
        "vocabulary_size": np.int32(vocabulary_size),
        "num_oov_buckets": np.int32(num_oov_buckets),
        "default_value": np.int32(default_value),
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    key = "products"
    vocabulary_file = vocab_file
    vocabulary_size = 5
    num_oov_buckets = 0
    default_value = -2
    dtype = tf.string

    input_dict = {
        "key": key,
        "vocabulary_file": vocabulary_file,
        "vocabulary_size": np.int32(vocabulary_size),
        "num_oov_buckets": np.int32(num_oov_buckets),
        "default_value": np.int32(default_value),
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 (smaller vocab size)
    key = "smaller_fruits"
    vocabulary_file = vocab_file
    vocabulary_size = 4
    num_oov_buckets = 0
    default_value = -1
    dtype = tf.string

    input_dict = {
        "key": key,
        "vocabulary_file": vocabulary_file,
        "vocabulary_size": np.int32(vocabulary_size),
        "num_oov_buckets": np.int32(num_oov_buckets),
        "default_value": np.int32(default_value),
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Clean up vocabulary file
    os.remove(vocab_file)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.sequence_categorical_column_with_vocabulary_file"] = tf_feature_column_sequence_categorical_column_with_vocabulary_file_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.sequence_categorical_column_with_vocabulary_file' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.sequence_categorical_column_with_vocabulary_file'.")

check_valid('tf.feature_column.sequence_categorical_column_with_vocabulary_file', generated_inputs['tf.feature_column.sequence_categorical_column_with_vocabulary_file'], lib="tf", suffix=0)
