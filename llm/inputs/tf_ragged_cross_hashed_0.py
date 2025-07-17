
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_cross_hashed_inputs():
    list_of_inputs = []

    # Input 1: Basic case with small integers
    inputs = [tf.ragged.constant([[1], [2, 3]]), tf.ragged.constant([[4], [5]])]
    num_buckets = 10
    hash_key = 123
    name = "basic_case"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Strings
    inputs = [tf.ragged.constant([['a'], ['b', 'c']]), tf.ragged.constant([['d'], ['e']])]
    num_buckets = 100
    hash_key = 456
    name = "strings"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero num_buckets
    inputs = [tf.ragged.constant([[1], [2, 3]]), tf.ragged.constant([[4], [5]])]
    num_buckets = 0
    hash_key = 789
    name = "zero_buckets"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensors instead of ragged tensors
    inputs = [tf.constant([[1], [2]]), tf.constant([[3], [4]])]
    num_buckets = 5
    hash_key = 101
    name = "tensors"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different lengths in ragged tensors, but valid number of rows
    inputs = [tf.ragged.constant([[1, 2], [3]]), tf.ragged.constant([[4], [5, 6, 7]])]
    num_buckets = 20
    hash_key = 202
    name = "different_lengths"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger numbers
    inputs = [tf.ragged.constant([[1000], [2000, 3000]]), tf.ragged.constant([[4000], [5000]])]
    num_buckets = 1000
    hash_key = 303
    name = "large_numbers"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: More inputs in the list
    inputs = [tf.ragged.constant([[1], [2]]), tf.ragged.constant([[3], [4]]), tf.ragged.constant([[5], [6]])]
    num_buckets = 30
    hash_key = 404
    name = "more_inputs"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Force Dense Tensors to be Ragged
    inputs = [tf.ragged.constant([[1], [2]]), tf.ragged.constant([[3], [4]])]
    num_buckets = 50
    hash_key = 606
    name = "mixed_types"
    input_dict = {"inputs": [tf.cast(inp, tf.string) if not isinstance(inp, tf.RaggedTensor) else inp for inp in inputs], "num_buckets": num_buckets, "hash_key": hash_key, "name": name}

    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: More complex strings
    inputs = [tf.ragged.constant([['hello'], ['world', 'again']]), tf.ragged.constant([['tensorflow'], ['is', 'cool']])]
    num_buckets = 10000
    hash_key = 707
    name = "complex_strings"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: UTF-8 characters
    inputs = [tf.ragged.constant([['你好'], ['世界']]), tf.ragged.constant([['TensorFlow'], ['真棒']])]
    num_buckets = 10000
    hash_key = 808
    name = "utf8_strings"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: int64 values
    inputs = [tf.ragged.constant([[1234567890], [9876543210]]), tf.ragged.constant([[1122334455], [5544332211]])]
    num_buckets = 100000
    hash_key = 909
    name = "int64_values"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.ragged.cross_hashed"] = tf_ragged_cross_hashed_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.ragged.cross_hashed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.cross_hashed'.")

check_valid('tf.ragged.cross_hashed', generated_inputs['tf.ragged.cross_hashed'], lib="tf", suffix=0)
