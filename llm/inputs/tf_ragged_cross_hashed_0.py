
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_cross_hashed_inputs():
    list_of_inputs = []

    # Input 1
    inputs = [tf.ragged.constant([['a'], ['b', 'c']]),
              tf.ragged.constant([['d'], ['e']]),
              tf.ragged.constant([['f'], ['g']])]
    num_buckets = 100
    hash_key = 123
    name = "cross_hashed_1"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = [tf.ragged.constant([['x', 'y'], ['z']]),
              tf.ragged.constant([['p'], ['q', 'r']])]
    num_buckets = 50
    hash_key = None
    name = "cross_hashed_2"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = [tf.ragged.constant([['1'], ['2', '3']]),
              tf.ragged.constant([['4'], ['5']]),
              tf.ragged.constant([['6'], ['7']])]
    num_buckets = 0
    hash_key = 456
    name = "cross_hashed_3"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = [tf.ragged.constant([['a']]),
              tf.ragged.constant([['b']]),
              tf.ragged.constant([['c']])]
    num_buckets = 10
    hash_key = None
    name = "cross_hashed_4"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = [tf.ragged.constant([['1', '2', '3'], ['4', '5']]),
              tf.ragged.constant([['6', '7'], ['8', '9']])]
    num_buckets = 1000
    hash_key = 789
    name = "cross_hashed_5"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Using tf.constant (Tensor) - Rectangular shape required
    inputs = [tf.constant([['a'], ['b']]),
              tf.constant([['c'], ['d']])]
    num_buckets = 2
    hash_key = 101
    name = "cross_hashed_6"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Mixed Tensor types
    inputs = [tf.ragged.constant([['a'], ['b']]),
              tf.constant([['c'], ['d']])]
    num_buckets = 4
    hash_key = 123
    name = "cross_hashed_8"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  Zero buckets
    inputs = [tf.ragged.constant([['a'], ['b']]),
              tf.ragged.constant([['c'], ['d']])]
    num_buckets = 0
    hash_key = 145
    name = "cross_hashed_10"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: More Ragged - keep the ragged tensors as consistent shape
    inputs = [tf.ragged.constant([['a', 'b'], ['c', 'd']]),
              tf.ragged.constant([['e', 'f'], ['g', 'h']])]
    num_buckets = 10
    hash_key = 987
    name = "cross_hashed_11"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different strings
    inputs = [tf.ragged.constant([['apple'], ['banana']]),
              tf.ragged.constant([['orange'], ['grape']])]
    num_buckets = 7
    hash_key = 42
    name = "cross_hashed_12"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Integer inputs
    inputs = [tf.ragged.constant([[1], [2]]),
              tf.ragged.constant([[3], [4]])]
    num_buckets = 11
    hash_key = 15
    name = "cross_hashed_13"
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
