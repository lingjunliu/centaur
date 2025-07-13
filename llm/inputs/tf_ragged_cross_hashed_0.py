
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_cross_hashed_inputs():
    list_of_inputs = []

    def to_numpy(x):
        if isinstance(x, tf.RaggedTensor):
            return x.to_list()
        elif isinstance(x, tf.Tensor):
            return x.numpy().tolist()
        else:
            return x

    # Input 1
    inputs = [tf.ragged.constant([['a'], ['b', 'c']]),
              tf.ragged.constant([['d'], ['e']]),
              tf.ragged.constant([['f'], ['g']])]
    num_buckets = 100
    hash_key = 123
    name = "example1"
    input_dict = {"inputs": [to_numpy(x) for x in inputs], "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = [tf.ragged.constant([['a', 'b'], ['c']]),
              tf.ragged.constant([['d'], ['e', 'f']])]
    num_buckets = 0
    hash_key = None
    name = None
    input_dict = {"inputs": [to_numpy(x) for x in inputs], "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = [tf.ragged.constant([['1'], ['2', '3']]),
              tf.ragged.constant([['4'], ['5']]),
              tf.ragged.constant([['6'], ['7']])]
    num_buckets = 5
    hash_key = 456
    name = "example3"
    input_dict = {"inputs": [to_numpy(x) for x in inputs], "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = [tf.ragged.constant([['a']]), tf.ragged.constant([['b']])]
    num_buckets = 10
    hash_key = None
    name = "example4"
    input_dict = {"inputs": [to_numpy(x) for x in inputs], "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = [tf.ragged.constant([['1', '2'], ['3', '4']]),
              tf.ragged.constant([['5', '6'], ['7', '8']])]
    num_buckets = 1000
    hash_key = 789
    name = "example5"
    input_dict = {"inputs": [to_numpy(x) for x in inputs], "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6 (using Tensors)
    inputs = [tf.constant([['a'], ['b']]),
              tf.constant([['c'], ['d']])]
    num_buckets = 20
    hash_key = 101
    name = "example6"
    input_dict = {"inputs": [to_numpy(x) for x in inputs], "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
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
