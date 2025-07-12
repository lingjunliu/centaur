
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_lookup_StaticVocabularyTable_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    keys = np.array(['apple', 'banana', 'cherry'])
    values = np.array([0, 1, 2], dtype=np.int64)
    num_oov_buckets = 5
    initializer = tf.lookup.KeyValueTensorInitializer(tf.constant(keys), tf.constant(values))
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: num_oov_buckets = 0
    keys = np.array(['dog', 'cat', 'mouse'])
    values = np.array([10, 11, 12], dtype=np.int64)
    num_oov_buckets = 0
    initializer = tf.lookup.KeyValueTensorInitializer(tf.constant(keys), tf.constant(values))
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger vocabulary and number of buckets
    keys = np.array([f'word_{i}' for i in range(10)])
    values = np.array(list(range(10)), dtype=np.int64)
    num_oov_buckets = 2
    initializer = tf.lookup.KeyValueTensorInitializer(tf.constant(keys), tf.constant(values))
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different value dtype
    keys = np.array(['one', 'two', 'three'])
    values = np.array([100, 200, 300], dtype=np.int64)
    num_oov_buckets = 1
    initializer = tf.lookup.KeyValueTensorInitializer(tf.constant(keys), tf.constant(values))
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty vocabulary
    keys = np.array([])
    values = np.array([], dtype=np.int64)
    num_oov_buckets = 3
    initializer = tf.lookup.KeyValueTensorInitializer(tf.constant(keys), tf.constant(values))
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Vocabulary with non-string keys (but still string)
    keys = np.array(['1', '2', '3'])
    values = np.array([1, 2, 3], dtype=np.int64)
    num_oov_buckets = 1
    initializer = tf.lookup.KeyValueTensorInitializer(tf.constant(keys), tf.constant(values))
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single item vocabulary
    keys = np.array(['solo'])
    values = np.array([42], dtype=np.int64)
    num_oov_buckets = 1
    initializer = tf.lookup.KeyValueTensorInitializer(tf.constant(keys), tf.constant(values))
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: Key with special characters
    keys = np.array(['!@#', '$%^', '&*()'])
    values = np.array([10, 11, 12], dtype=np.int64)
    num_oov_buckets = 1
    initializer = tf.lookup.KeyValueTensorInitializer(tf.constant(keys), tf.constant(values))
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Unicode keys
    keys = np.array(['你好', '世界', '你好世界'])
    values = np.array([0, 1, 2], dtype=np.int64)
    num_oov_buckets = 1
    initializer = tf.lookup.KeyValueTensorInitializer(tf.constant(keys), tf.constant(values))
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: num_oov_buckets=1
    keys = np.array(['a', 'b', 'c'])
    values = np.array([0, 1, 2], dtype=np.int64)
    num_oov_buckets = 1
    initializer = tf.lookup.KeyValueTensorInitializer(tf.constant(keys), tf.constant(values))
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
