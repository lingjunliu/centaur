
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_cross_hashed_inputs():
    list_of_inputs = []

    # Input 1: Basic example with RaggedTensors
    inputs = [tf.ragged.constant([['a'], ['b', 'c']]),
              tf.ragged.constant([['d'], ['e']]),
              tf.ragged.constant([['f'], ['g']])
              ]
    num_buckets = 100
    hash_key = 12345
    name = "basic_example"
    input_dict = {'inputs': inputs, 'num_buckets': num_buckets, 'hash_key': hash_key, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With regular Tensors
    inputs = [tf.constant([['a'], ['b']]),
              tf.constant([['c'], ['d']]),
              tf.constant([['e'], ['f']])
              ]
    num_buckets = 50
    hash_key = 67890
    name = "regular_tensors"
    input_dict = {'inputs': inputs, 'num_buckets': num_buckets, 'hash_key': hash_key, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With SparseTensors
    inputs = [tf.sparse.from_dense(tf.constant([['a'], ['b']])),
              tf.sparse.from_dense(tf.constant([['c'], ['d']])),
              tf.sparse.from_dense(tf.constant([['e'], ['f']]))
              ]
    num_buckets = 25
    hash_key = 13579
    name = "sparse_tensors"
    input_dict = {'inputs': inputs, 'num_buckets': num_buckets, 'hash_key': hash_key, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zero num_buckets
    inputs = [tf.ragged.constant([['a'], ['b', 'c']]),
              tf.ragged.constant([['d'], ['e']]),
              tf.ragged.constant([['f'], ['g']])
              ]
    num_buckets = 0
    hash_key = 24680
    name = "zero_buckets"
    input_dict = {'inputs': inputs, 'num_buckets': num_buckets, 'hash_key': hash_key, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different string lengths
    inputs = [tf.ragged.constant([['a'], ['bc']]),
              tf.ragged.constant([['def'], ['g']]),
              tf.ragged.constant([['hi'], ['jkl']])
              ]
    num_buckets = 75
    hash_key = 91234
    name = "different_lengths"
    input_dict = {'inputs': inputs, 'num_buckets': num_buckets, 'hash_key': hash_key, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: More inputs in the list
    inputs = [tf.ragged.constant([['a'], ['b']]),
              tf.ragged.constant([['c'], ['d']]),
              tf.ragged.constant([['e'], ['f']]),
              tf.ragged.constant([['g'], ['h']])
              ]
    num_buckets = 10
    hash_key = 56789
    name = "more_inputs"
    input_dict = {'inputs': inputs, 'num_buckets': num_buckets, 'hash_key': hash_key, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty strings
    inputs = [tf.ragged.constant([[''], ['b']]),
              tf.ragged.constant([['c'], ['']]),
              tf.ragged.constant([['e'], ['f']])
              ]
    num_buckets = 30
    hash_key = 11223
    name = "empty_strings"
    input_dict = {'inputs': inputs, 'num_buckets': num_buckets, 'hash_key': hash_key, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single character strings
    inputs = [tf.ragged.constant([['x'], ['y']]),
              tf.ragged.constant([['z'], ['w']]),
              tf.ragged.constant([['u'], ['v']])
              ]
    num_buckets = 60
    hash_key = 33445
    name = "single_chars"
    input_dict = {'inputs': inputs, 'num_buckets': num_buckets, 'hash_key': hash_key, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Same strings repeated
    inputs = [tf.ragged.constant([['a'], ['a']]),
              tf.ragged.constant([['b'], ['b']]),
              tf.ragged.constant([['c'], ['c']])
              ]
    num_buckets = 40
    hash_key = 55667
    name = "repeated_strings"
    input_dict = {'inputs': inputs, 'num_buckets': num_buckets, 'hash_key': hash_key, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Combination of all input types
    inputs = [tf.ragged.constant([['a'], ['b', 'c']]),
              tf.constant([['d'], ['e']]),
              tf.sparse.from_dense(tf.constant([['f'], ['g']]))
              ]
    num_buckets = 80
    hash_key = 77889
    name = "mixed_types"
    input_dict = {'inputs': inputs, 'num_buckets': num_buckets, 'hash_key': hash_key, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: RaggedTensor of integers
    inputs = [tf.ragged.constant([[1], [2, 3]]),
              tf.ragged.constant([[4], [5]]),
              tf.ragged.constant([[6], [7]])
              ]
    num_buckets = 100
    hash_key = 12345
    name = "int_example"
    input_dict = {'inputs': inputs, 'num_buckets': num_buckets, 'hash_key': hash_key, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: RaggedTensor of floats
    inputs = [tf.ragged.constant([[1.0], [2.5, 3.0]]),
              tf.ragged.constant([[4.0], [5.0]]),
              tf.ragged.constant([[6.0], [7.5]])
              ]
    num_buckets = 100
    hash_key = 12345
    name = "float_example"
    input_dict = {'inputs': inputs, 'num_buckets': num_buckets, 'hash_key': hash_key, 'name': name}
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
