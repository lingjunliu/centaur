
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_lookup_StaticVocabularyTable_inputs():
    list_of_inputs = []

    # Input 1: Simple case with string keys and int values
    keys = np.array(['a', 'b', 'c'])
    values = np.array([0, 1, 2], dtype=np.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(keys=tf.convert_to_tensor(keys), values=tf.convert_to_tensor(values))
    num_oov_buckets = 2
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Empty initializer
    initializer = None
    num_oov_buckets = 5
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger vocabulary and more OOV buckets
    keys = np.array(['apple', 'banana', 'cherry', 'date', 'fig'])
    values = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(keys=tf.convert_to_tensor(keys), values=tf.convert_to_tensor(values))
    num_oov_buckets = 10
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Single OOV bucket
    keys = np.array(['cat', 'dog'])
    values = np.array([1, 2], dtype=np.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(keys=tf.convert_to_tensor(keys), values=tf.convert_to_tensor(values))
    num_oov_buckets = 1
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero OOV buckets (not recommended, but valid)
    keys = np.array(['one', 'two', 'three'])
    values = np.array([100, 200, 300], dtype=np.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(keys=tf.convert_to_tensor(keys), values=tf.convert_to_tensor(values))
    num_oov_buckets = 0
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Only OOV buckets.
    initializer = None
    num_oov_buckets = 3
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger num_oov_buckets
    keys = np.array(['x', 'y', 'z'])
    values = np.array([7, 8, 9], dtype=np.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(keys=tf.convert_to_tensor(keys), values=tf.convert_to_tensor(values))
    num_oov_buckets = 15
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different keys
    keys = np.array(['hello', 'world', '!'])
    values = np.array([4, 5, 6], dtype=np.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(keys=tf.convert_to_tensor(keys), values=tf.convert_to_tensor(values))
    num_oov_buckets = 4
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: More keys and values
    keys = np.array(['a1', 'b2', 'c3', 'd4', 'e5', 'f6'])
    values = np.array([11, 22, 33, 44, 55, 66], dtype=np.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(keys=tf.convert_to_tensor(keys), values=tf.convert_to_tensor(values))
    num_oov_buckets = 7
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger table with initializer
    keys = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    values = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(keys=tf.convert_to_tensor(keys), values=tf.convert_to_tensor(values))
    num_oov_buckets = 5
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.lookup.StaticVocabularyTable"] = tf_lookup_StaticVocabularyTable_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.lookup.StaticVocabularyTable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.lookup.StaticVocabularyTable'.")

check_valid('tf.lookup.StaticVocabularyTable', generated_inputs['tf.lookup.StaticVocabularyTable'], lib="tf", suffix=0)
