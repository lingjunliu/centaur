
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_lookup_StaticVocabularyTable_inputs():
    list_of_inputs = []

    # Input 1
    keys = np.array(['a', 'b', 'c'])
    values = np.array([0, 1, 2], dtype=np.int64)
    num_oov_buckets = 3
    input_dict = {"initializer": tf.constant([keys, values]), "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    keys = np.array(['x', 'y', 'z', 'w'])
    values = np.array([10, 20, 30, 40], dtype=np.int64)
    num_oov_buckets = 5
    input_dict = {"initializer": tf.constant([keys, values]), "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    keys = np.array(['one', 'two'])
    values = np.array([1, 2], dtype=np.int64)
    num_oov_buckets = 1
    input_dict = {"initializer": tf.constant([keys, values]), "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    keys = np.array([])
    values = np.array([], dtype=np.int64)
    num_oov_buckets = 10
    input_dict = {"initializer": tf.constant([keys, values]), "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    keys = np.array(['apple', 'banana', 'cherry', 'date', 'elderberry'])
    values = np.array([100, 200, 300, 400, 500], dtype=np.int64)
    num_oov_buckets = 0
    input_dict = {"initializer": tf.constant([keys, values]), "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    keys = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    values = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    num_oov_buckets = 7
    input_dict = {"initializer": tf.constant([keys, values]), "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    keys = np.array(['A', 'B'])
    values = np.array([65, 66], dtype=np.int64)
    num_oov_buckets = 2
    input_dict = {"initializer": tf.constant([keys, values]), "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    keys = np.array(['one', 'two', 'three', 'four'])
    values = np.array([1, 2, 3, 4], dtype=np.int64)
    num_oov_buckets = 4
    input_dict = {"initializer": tf.constant([keys, values]), "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty keys and values, zero oov buckets
    keys = np.array([])
    values = np.array([], dtype=np.int64)
    num_oov_buckets = 0
    input_dict = {"initializer": tf.constant([keys, values]), "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: More oov buckets than keys
    keys = np.array(['word1', 'word2'])
    values = np.array([1000, 2000], dtype=np.int64)
    num_oov_buckets = 10
    input_dict = {"initializer": tf.constant([keys, values]), "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.lookup.StaticVocabularyTable"] = tf_lookup_StaticVocabularyTable_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.lookup.StaticVocabularyTable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.lookup.StaticVocabularyTable'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.lookup.StaticVocabularyTable', generated_inputs['tf.lookup.StaticVocabularyTable'], lib="tf", suffix=0)
