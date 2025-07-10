
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_lookup_static_vocabulary_table_inputs():
    list_of_inputs = []

    # Input 1
    keys = np.array(['a', 'b', 'c'])
    values = np.array([0, 1, 2], dtype=np.int64)
    init = tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(keys),
        values=tf.constant(values)
    )
    num_oov_buckets = np.int32(1)
    input_dict = {"initializer": init, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    keys = np.array(['a', 'b', 'c'])
    values = np.array([10, 20, 30], dtype=np.int64)
    init = tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(keys),
        values=tf.constant(values)
    )
    num_oov_buckets = np.int32(5)
    input_dict = {"initializer": init, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    keys = np.array(['one', 'two', 'three', 'four'])
    values = np.array([1, 2, 3, 4], dtype=np.int64)
    init = tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(keys),
        values=tf.constant(values)
    )
    num_oov_buckets = np.int32(0)
    input_dict = {"initializer": init, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    keys = np.array(['red', 'green', 'blue'])
    values = np.array([0, -1, -2], dtype=np.int64)
    init = tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(keys),
        values=tf.constant(values)
    )
    num_oov_buckets = np.int32(2)
    input_dict = {"initializer": init, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    keys = np.array(['x', 'y', 'z', 'w'])
    values = np.array([100, 200, 300, 400], dtype=np.int64)
    init = tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(keys),
        values=tf.constant(values)
    )
    num_oov_buckets = np.int32(10)
    input_dict = {"initializer": init, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    keys = np.array(['apple', 'banana'])
    values = np.array([5, 6], dtype=np.int64)
    init = tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(keys),
        values=tf.constant(values)
    )
    num_oov_buckets = np.int32(3)
    input_dict = {"initializer": init, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger vocabulary size
    keys = np.array([str(i) for i in range(100)])
    values = np.array(list(range(100)), dtype=np.int64)
    init = tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(keys),
        values=tf.constant(values)
    )
    num_oov_buckets = np.int32(1)
    input_dict = {"initializer": init, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: Init with empty vocabulary
    keys = np.array([], dtype=np.string_)
    values = np.array([], dtype=np.int64)
    init = tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(keys),
        values=tf.constant(values)
    )
    num_oov_buckets = np.int32(5)
    input_dict = {"initializer": init, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large Number of OOV Buckets
    keys = np.array(['cat', 'dog'])
    values = np.array([7, 8], dtype=np.int64)
    init = tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(keys),
        values=tf.constant(values)
    )
    num_oov_buckets = np.int32(100)
    input_dict = {"initializer": init, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    keys = np.array(['aa', 'bb', 'cc', 'dd', 'ee'])
    values = np.array([11, 22, 33, 44, 55], dtype=np.int64)
    init = tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(keys),
        values=tf.constant(values)
    )
    num_oov_buckets = np.int32(4)
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
