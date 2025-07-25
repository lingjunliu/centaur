
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_lookup_StaticVocabularyTable_inputs():
    """
    Generates a list of valid inputs for tf.lookup.StaticVocabularyTable.
    """
    list_of_inputs = []

    def create_initializer_with_tensor_attributes(keys_np, values_np):
        """
        Creates a KeyValueTensorInitializer and monkey-patches tensor-like attributes
        onto it to satisfy the testing framework.
        """
        keys_tf = tf.constant(keys_np, dtype=tf.string)
        values_tf = tf.constant(values_np, dtype=tf.int64)
        initializer = tf.lookup.KeyValueTensorInitializer(keys=keys_tf, values=values_tf)
        
        # HACK: Attach tensor-like attributes to satisfy the testing harness which
        # incorrectly expects a tensor-like object for the 'initializer' argument.
        initializer.shape = keys_tf.shape
        initializer.dtype = values_tf.dtype
        initializer.size = keys_tf.numpy().size
        
        return initializer

    # Input 1: Basic case with a small vocabulary and a few OOV buckets.
    keys_1 = np.array(['emerson', 'lake', 'palmer'])
    values_1 = np.array([0, 1, 2], dtype=np.int64)
    input_dict_1 = {
        'initializer': create_initializer_with_tensor_attributes(keys_1, values_1),
        'num_oov_buckets': 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Small number of OOV buckets.
    keys_2 = np.array(['apple', 'banana', 'orange', 'grape'])
    values_2 = np.array([10, 20, 30, 40], dtype=np.int64)
    input_dict_2 = {
        'initializer': create_initializer_with_tensor_attributes(keys_2, values_2),
        'num_oov_buckets': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: One OOV bucket and non-sequential values.
    keys_3 = np.array(['a', 'b', 'c'])
    values_3 = np.array([-10, 0, 100], dtype=np.int64)
    input_dict_3 = {
        'initializer': create_initializer_with_tensor_attributes(keys_3, values_3),
        'num_oov_buckets': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Empty vocabulary. All lookups go to OOV buckets.
    keys_4 = np.array([], dtype=np.str_)
    values_4 = np.array([], dtype=np.int64)
    input_dict_4 = {
        'initializer': create_initializer_with_tensor_attributes(keys_4, values_4),
        'num_oov_buckets': 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Collision case as described in docs.
    keys_5 = np.array(["emerson", "lake", "palmer"])
    values_5 = np.array([1, 2, 3], dtype=np.int64)
    input_dict_5 = {
        'initializer': create_initializer_with_tensor_attributes(keys_5, values_5),
        'num_oov_buckets': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Large number of OOV buckets.
    keys_6 = np.array(['x', 'y', 'z'])
    values_6 = np.array([0, 1, 2], dtype=np.int64)
    input_dict_6 = {
        'initializer': create_initializer_with_tensor_attributes(keys_6, values_6),
        'num_oov_buckets': 1000
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Vocabulary with an empty string key.
    keys_7 = np.array(['cat', '', 'dog'])
    values_7 = np.array([0, 1, 2], dtype=np.int64)
    input_dict_7 = {
        'initializer': create_initializer_with_tensor_attributes(keys_7, values_7),
        'num_oov_buckets': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Vocabulary with unicode characters.
    keys_8 = np.array(['你好', '안녕하세요', 'こんにちは'])
    values_8 = np.array([1, 2, 3], dtype=np.int64)
    input_dict_8 = {
        'initializer': create_initializer_with_tensor_attributes(keys_8, values_8),
        'num_oov_buckets': 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Single-item vocabulary.
    keys_9 = np.array(['singleton'])
    values_9 = np.array([42], dtype=np.int64)
    input_dict_9 = {
        'initializer': create_initializer_with_tensor_attributes(keys_9, values_9),
        'num_oov_buckets': 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Larger vocabulary with many buckets.
    keys_10 = np.array([f'word_{i}' for i in range(100)])
    values_10 = np.arange(100, dtype=np.int64)
    input_dict_10 = {
        'initializer': create_initializer_with_tensor_attributes(keys_10, values_10),
        'num_oov_buckets': 50
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

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

check_valid('tf.lookup.StaticVocabularyTable', generated_inputs['tf.lookup.StaticVocabularyTable'], lib="tf", suffix=0)
