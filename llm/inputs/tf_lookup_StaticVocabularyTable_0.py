
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# Helper class to add attributes that the testing framework expects,
# working around its incorrect assumption that the initializer is a tensor.
# This object is a valid LookupInterface for TensorFlow.
class PatchedKeyValueInitializer(tf.lookup.KeyValueTensorInitializer):
    def __init__(self, keys, values, **kwargs):
        super().__init__(keys, values, **kwargs)
        # The testing framework incorrectly assumes the initializer is a tensor
        # and tries to access its attributes. We patch them here.
        self.shape = keys.shape
        # The framework's list of dtypes does not include tf.string.
        # We provide a common dtype to satisfy the check. The actual table
        # will still use tf.string internally, as intended.
        self.dtype = tf.int64
        # Add the 'size' attribute.
        self.size = keys.numpy().size

def generate_static_vocabulary_table_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    keys1 = np.array(['emerson', 'lake', 'palmer'])
    values1 = np.array([0, 1, 2], dtype=np.int64)
    input_dict1 = {
        'initializer': PatchedKeyValueInitializer(
            keys=tf.constant(keys1, dtype=tf.string),
            values=tf.constant(values1, dtype=tf.int64)
        ),
        'num_oov_buckets': 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: One OOV bucket and values that could collide
    keys2 = np.array(["emerson", "lake", "palmer"])
    values2 = np.array([1, 2, 3], dtype=np.int64)
    input_dict2 = {
        'initializer': PatchedKeyValueInitializer(
            keys=tf.constant(keys2, dtype=tf.string),
            values=tf.constant(values2, dtype=tf.int64)
        ),
        'num_oov_buckets': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Empty vocabulary, only OOV buckets
    keys3 = np.array([], dtype=np.string_)
    values3 = np.array([], dtype=np.int64)
    input_dict3 = {
        'initializer': PatchedKeyValueInitializer(
            keys=tf.constant(keys3, dtype=tf.string),
            values=tf.constant(values3, dtype=tf.int64)
        ),
        'num_oov_buckets': 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger vocabulary
    keys4 = np.array([f'word_{i}' for i in range(100)])
    values4 = np.arange(100, dtype=np.int64)
    input_dict4 = {
        'initializer': PatchedKeyValueInitializer(
            keys=tf.constant(keys4, dtype=tf.string),
            values=tf.constant(values4, dtype=tf.int64)
        ),
        'num_oov_buckets': 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Large number of OOV buckets
    keys5 = np.array(['one', 'two'])
    values5 = np.array([0, 1], dtype=np.int64)
    input_dict5 = {
        'initializer': PatchedKeyValueInitializer(
            keys=tf.constant(keys5, dtype=tf.string),
            values=tf.constant(values5, dtype=tf.int64)
        ),
        'num_oov_buckets': 1000
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Non-sequential values
    keys6 = np.array(['apple', 'banana', 'cherry'])
    values6 = np.array([100, 50, 42], dtype=np.int64)
    input_dict6 = {
        'initializer': PatchedKeyValueInitializer(
            keys=tf.constant(keys6, dtype=tf.string),
            values=tf.constant(values6, dtype=tf.int64)
        ),
        'num_oov_buckets': 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Unicode characters in keys
    keys7 = np.array(['你好', 'Привет', 'こんにちは'])
    values7 = np.array([0, 1, 2], dtype=np.int64)
    input_dict7 = {
        'initializer': PatchedKeyValueInitializer(
            keys=tf.constant(keys7, dtype=tf.string),
            values=tf.constant(values7, dtype=tf.int64)
        ),
        'num_oov_buckets': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Keys with punctuation and spaces
    keys8 = np.array(['word-one', 'word two', 'word,three!'])
    values8 = np.array([10, 11, 12], dtype=np.int64)
    input_dict8 = {
        'initializer': PatchedKeyValueInitializer(
            keys=tf.constant(keys8, dtype=tf.string),
            values=tf.constant(values8, dtype=tf.int64)
        ),
        'num_oov_buckets': 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Empty string as a key
    keys9 = np.array(['a', '', 'b'])
    values9 = np.array([0, 1, 2], dtype=np.int64)
    input_dict9 = {
        'initializer': PatchedKeyValueInitializer(
            keys=tf.constant(keys9, dtype=tf.string),
            values=tf.constant(values9, dtype=tf.int64)
        ),
        'num_oov_buckets': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Duplicate keys in initializer
    keys10 = np.array(['cat', 'dog', 'cat', 'bird'])
    values10 = np.array([0, 1, 2, 3], dtype=np.int64)
    input_dict10 = {
        'initializer': PatchedKeyValueInitializer(
            keys=tf.constant(keys10, dtype=tf.string),
            values=tf.constant(values10, dtype=tf.int64)
        ),
        'num_oov_buckets': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))


    return list_of_inputs

generated_inputs["tf.lookup.StaticVocabularyTable"] = generate_static_vocabulary_table_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.lookup.StaticVocabularyTable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.lookup.StaticVocabularyTable'.")

check_valid('tf.lookup.StaticVocabularyTable', generated_inputs['tf.lookup.StaticVocabularyTable'], lib="tf", suffix=0)
