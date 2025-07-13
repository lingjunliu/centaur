
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_lookup_static_vocabulary_table_inputs():
    list_of_inputs = []

    # Input 1
    keys_tensor = tf.constant(np.array(['a', 'b', 'c']), dtype=tf.string)
    values_tensor = tf.constant(np.array([0, 1, 2], dtype=np.int64))
    init = tf.lookup.KeyValueTensorInitializer(keys=keys_tensor, values=values_tensor)
    num_oov_buckets = 1
    input_dict = {"initializer": init, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    keys_tensor = tf.constant(np.array(['a', 'b', 'c']), dtype=tf.string)
    values_tensor = tf.constant(np.array([10, 20, 30], dtype=np.int64))
    init = tf.lookup.KeyValueTensorInitializer(keys=keys_tensor, values=values_tensor)
    num_oov_buckets = 5
    input_dict = {"initializer": init, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 - Empty initializer
    keys_tensor = tf.constant(np.array([]), dtype=tf.string)
    values_tensor = tf.constant(np.array([], dtype=np.int64))
    init = tf.lookup.KeyValueTensorInitializer(keys=keys_tensor, values=values_tensor)
    num_oov_buckets = 3
    input_dict = {"initializer": init, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 - Larger num_oov_buckets
    keys_tensor = tf.constant(np.array(['x', 'y']), dtype=tf.string)
    values_tensor = tf.constant(np.array([100, 200], dtype=np.int64))
    init = tf.lookup.KeyValueTensorInitializer(keys=keys_tensor, values=values_tensor)
    num_oov_buckets = 10
    input_dict = {"initializer": init, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 - More keys
    keys_tensor = tf.constant(np.array(['a', 'b', 'c', 'd', 'e']), dtype=tf.string)
    values_tensor = tf.constant(np.array([0, 1, 2, 3, 4], dtype=np.int64))
    init = tf.lookup.KeyValueTensorInitializer(keys=keys_tensor, values=values_tensor)
    num_oov_buckets = 2
    input_dict = {"initializer": init, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - num_oov_buckets = 0
    keys_tensor = tf.constant(np.array(['p', 'q', 'r']), dtype=tf.string)
    values_tensor = tf.constant(np.array([5, 6, 7], dtype=np.int64))
    init = tf.lookup.KeyValueTensorInitializer(keys=keys_tensor, values=values_tensor)
    num_oov_buckets = 0
    input_dict = {"initializer": init, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Different values
    keys_tensor = tf.constant(np.array(['s', 't', 'u']), dtype=tf.string)
    values_tensor = tf.constant(np.array([15, 25, 35], dtype=np.int64))
    init = tf.lookup.KeyValueTensorInitializer(keys=keys_tensor, values=values_tensor)
    num_oov_buckets = 4
    input_dict = {"initializer": init, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - More oov buckets than keys
    keys_tensor = tf.constant(np.array(['one', 'two']), dtype=tf.string)
    values_tensor = tf.constant(np.array([11, 22], dtype=np.int64))
    init = tf.lookup.KeyValueTensorInitializer(keys=keys_tensor, values=values_tensor)
    num_oov_buckets = 7
    input_dict = {"initializer": init, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.lookup.StaticVocabularyTable"] = tf_lookup_static_vocabulary_table_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.lookup.StaticVocabularyTable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.lookup.StaticVocabularyTable'.")

check_valid('tf.lookup.StaticVocabularyTable', generated_inputs['tf.lookup.StaticVocabularyTable'], lib="tf", suffix=0)
