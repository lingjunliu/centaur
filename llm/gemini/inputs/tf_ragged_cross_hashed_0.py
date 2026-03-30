
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
    name = "test_cross_hashed_1"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = [tf.ragged.constant([['a', 'b'], ['c']]),
              tf.ragged.constant([['d'], ['e', 'f']])]
    num_buckets = 0
    hash_key = None
    name = None
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = [tf.ragged.constant([['1'], ['2', '3']]),
              tf.ragged.constant([['4'], ['5']]),
              tf.ragged.constant([['6'], ['7']])
              ]
    num_buckets = 50
    hash_key = 456
    name = "test_cross_hashed_3"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = [tf.ragged.constant([['a']]), tf.ragged.constant([['b']])]
    num_buckets = 10
    hash_key = 789
    name = "test_cross_hashed_4"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = [tf.ragged.constant([['x', 'y', 'z'], ['p', 'q']]), tf.ragged.constant([['a', 'b'], ['c']])]
    num_buckets = 1000
    hash_key = 101112
    name = "test_cross_hashed_5"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = [tf.ragged.constant([['1'], ['2']]), tf.ragged.constant([['3'], ['4']]), tf.ragged.constant([['5'], ['6']])]
    num_buckets = 1
    hash_key = 131415
    name = "test_cross_hashed_6"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = [tf.ragged.constant([['']]), tf.ragged.constant([['']])]
    num_buckets = 100
    hash_key = 161718
    name = "test_cross_hashed_7"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor inputs
    inputs = [tf.constant([['a'], ['b']]), tf.constant([['c'], ['d']])]
    num_buckets = 5
    hash_key = 192021
    name = "test_cross_hashed_8"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: RaggedTensor and Tensor
    inputs = [tf.ragged.constant([['a'], ['b']]), tf.constant([['c'], ['d']])]
    num_buckets = 10
    hash_key = 252627
    name = "test_cross_hashed_10"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10: All Tensor inputs. Ensure they have the same dtype
    inputs = [tf.constant([['a'], ['b']], dtype=tf.string),
              tf.constant([['c'], ['d']], dtype=tf.string)]
    num_buckets = 7
    hash_key = 222324
    name = "test_cross_hashed_9"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.ragged.cross_hashed"] = tf_ragged_cross_hashed_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.ragged.cross_hashed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.cross_hashed'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.ragged.cross_hashed', generated_inputs['tf.ragged.cross_hashed'], lib="tf", suffix=0)
