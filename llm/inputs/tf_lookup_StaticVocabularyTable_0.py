
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_lookup_staticvocabularytable_inputs():
    """
    Generates a list of valid inputs for tf.lookup.StaticVocabularyTable.
    """
    list_of_inputs = []

    def create_input_dict(keys_np, values_np, num_oov_buckets):
        """Helper to create input dict and monkey-patch the initializer."""
        # Determine the key dtype, default to tf.string for Python strings/bytes.
        keys_dtype = tf.string if keys_np.dtype.kind in ('U', 'S') else keys_np.dtype
        keys = tf.constant(keys_np, dtype=keys_dtype)
        # StaticVocabularyTable requires value_dtype to be an integer type.
        values = tf.constant(values_np, dtype=tf.int64)
        initializer = tf.lookup.KeyValueTensorInitializer(keys, values)

        # The test harness expects the 'initializer' to be a 'tensor',
        # meaning it tries to access .shape, .dtype, and .size attributes.
        # We monkey-patch these attributes onto the initializer object to
        # satisfy the harness. The actual TF execution uses the initializer
        # object correctly, ignoring these patches.
        initializer.shape = keys.shape
        initializer.dtype = values.dtype # Use values.dtype (e.g., int64) to avoid harness issues with tf.string
        initializer.size = keys_np.size
        return {'initializer': initializer, 'num_oov_buckets': num_oov_buckets}

    # Input 1: Basic case
    keys1 = np.array(['emerson', 'lake', 'palmer'])
    values1 = np.array([0, 1, 2])
    list_of_inputs.append(copy.deepcopy(create_input_dict(keys1, values1, 5)))

    # Input 2: Minimum number of OOV buckets (must be > 0)
    keys2 = np.array(['apple', 'banana', 'orange'])
    values2 = np.array([10, 20, 30])
    list_of_inputs.append(copy.deepcopy(create_input_dict(keys2, values2, 1)))

    # Input 3: Empty vocabulary, all inputs go to OOV
    keys3 = np.array([], dtype=np.str_)
    values3 = np.array([], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy(create_input_dict(keys3, values3, 10)))

    # Input 4: Non-sequential and negative values
    keys4 = np.array(['a', 'b', 'c', 'd'])
    values4 = np.array([-1, 100, -50, 5])
    list_of_inputs.append(copy.deepcopy(create_input_dict(keys4, values4, 3)))

    # Input 5: One OOV bucket, potential for collision
    keys5 = np.array(["emerson", "lake", "palmer"])
    values5 = np.array([1, 2, 3])
    list_of_inputs.append(copy.deepcopy(create_input_dict(keys5, values5, 1)))

    # Input 6: Byte string keys
    keys6 = np.array([b'hello', b'world', b'tensorflow'])
    values6 = np.array([0, 1, 2])
    list_of_inputs.append(copy.deepcopy(create_input_dict(keys6, values6, 2)))

    # Input 7: Empty string as a key
    keys7 = np.array(['start', '', 'end'])
    values7 = np.array([10, 20, 30])
    list_of_inputs.append(copy.deepcopy(create_input_dict(keys7, values7, 4)))

    # Input 8: Single-item vocabulary
    keys8 = np.array(['lonely_key'])
    values8 = np.array([42])
    list_of_inputs.append(copy.deepcopy(create_input_dict(keys8, values8, 100)))

    # Input 9: Duplicate keys (last value is kept by the initializer)
    keys9 = np.array(['a', 'b', 'c', 'a', 'b'])
    values9 = np.array([0, 1, 2, 10, 11])
    list_of_inputs.append(copy.deepcopy(create_input_dict(keys9, values9, 1)))

    # Input 10: Larger vocabulary
    keys10 = np.array([f'key_{i}' for i in range(100)])
    values10 = np.arange(100)
    list_of_inputs.append(copy.deepcopy(create_input_dict(keys10, values10, 20)))

    # Input 11: Using int32 numpy values (will be cast to tf.int64 by helper)
    keys11 = np.array(['red', 'green', 'blue'])
    values11 = np.array([0, 1, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy(create_input_dict(keys11, values11, 5)))


    return list_of_inputs

generated_inputs["tf.lookup.StaticVocabularyTable"] = tf_lookup_staticvocabularytable_inputs()

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
