
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_lookup_static_vocabulary_table_inputs():
    list_of_inputs = []

    # Input 1: Basic case with strings and ints
    init_keys = np.array(['apple', 'banana', 'cherry'])
    init_values = np.array([0, 1, 2], dtype=np.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(keys=tf.constant(init_keys), values=tf.constant(init_values))
    num_oov_buckets = 3
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: No initial values, only OOV buckets
    initializer = None
    num_oov_buckets = 5
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different number of OOV buckets
    init_keys = np.array(['cat', 'dog'])
    init_values = np.array([0, 1], dtype=np.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(keys=tf.constant(init_keys), values=tf.constant(init_values))
    num_oov_buckets = 10
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zero OOV buckets (not recommended but valid)
    init_keys = np.array(['one', 'two', 'three'])
    init_values = np.array([0, 1, 2], dtype=np.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(keys=tf.constant(init_keys), values=tf.constant(init_values))
    num_oov_buckets = 0
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: More initial values than OOV buckets
    init_keys = np.array(['red', 'green', 'blue', 'yellow', 'purple'])
    init_values = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(keys=tf.constant(init_keys), values=tf.constant(init_values))
    num_oov_buckets = 2
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Empty vocabulary with oov buckets
    init_keys = np.array([])
    init_values = np.array([], dtype=np.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(keys=tf.constant(init_keys), values=tf.constant(init_values))
    num_oov_buckets = 5
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger number of OOV buckets
    init_keys = np.array(['car', 'truck'])
    init_values = np.array([0, 1], dtype=np.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(keys=tf.constant(init_keys), values=tf.constant(init_values))
    num_oov_buckets = 50
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: With longer strings
    init_keys = np.array(['the quick brown fox', 'jumps over the lazy dog'])
    init_values = np.array([0, 1], dtype=np.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(keys=tf.constant(init_keys), values=tf.constant(init_values))
    num_oov_buckets = 3
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Unicode characters
    init_keys = np.array(['你好', '世界'])
    init_values = np.array([0, 1], dtype=np.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(keys=tf.constant(init_keys), values=tf.constant(init_values))
    num_oov_buckets = 2
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Empty String Keys and Initial Values
    init_keys = np.array(['', 'a'])
    init_values = np.array([0, 1], dtype=np.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(keys=tf.constant(init_keys), values=tf.constant(init_values))
    num_oov_buckets = 1
    input_dict = {"initializer": initializer, "num_oov_buckets": num_oov_buckets}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
def to_numpy(v):
    if isinstance(v, tf.Tensor):
        return v.numpy()
    return v

def get_shape(v):
  if v is None:
    return []
  if isinstance(v, tf.lookup.KeyValueTensorInitializer):
      return [1]
  return list(to_numpy(v).shape)

generated_inputs["tf.lookup.StaticVocabularyTable"] = tf_lookup_static_vocabulary_table_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.lookup.StaticVocabularyTable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.lookup.StaticVocabularyTable'.")

check_valid('tf.lookup.StaticVocabularyTable', generated_inputs['tf.lookup.StaticVocabularyTable'], lib="tf", suffix=0)
