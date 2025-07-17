
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_cross_hashed_inputs():
    list_of_inputs = []

    # Input 1: Basic example with two SparseTensors
    st1 = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=np.array(['a', 'b'], dtype=np.string_), dense_shape=[2, 1])
    st2 = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=np.array(['c', 'd'], dtype=np.string_), dense_shape=[2, 1])
    inputs = [st1, st2]
    num_buckets = 0
    hash_key = None
    name = None

    input_dict = {
        "inputs": inputs,
        "num_buckets": num_buckets,
        "hash_key": hash_key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With num_buckets
    st1 = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=np.array(['a', 'b'], dtype=np.string_), dense_shape=[2, 1])
    st2 = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=np.array(['c', 'd'], dtype=np.string_), dense_shape=[2, 1])
    inputs = [st1, st2]
    num_buckets = 100
    hash_key = None
    name = None

    input_dict = {
        "inputs": inputs,
        "num_buckets": num_buckets,
        "hash_key": hash_key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With hash_key
    st1 = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=np.array(['a', 'b'], dtype=np.string_), dense_shape=[2, 1])
    st2 = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=np.array(['c', 'd'], dtype=np.string_), dense_shape=[2, 1])
    inputs = [st1, st2]
    num_buckets = 0
    hash_key = 12345
    name = None

    input_dict = {
        "inputs": inputs,
        "num_buckets": num_buckets,
        "hash_key": hash_key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With a dense tensor
    st1 = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=np.array(['a', 'b'], dtype=np.string_), dense_shape=[2, 1])
    t2 = tf.constant([['c'], ['d']], dtype=tf.string)
    inputs = [st1, t2]
    num_buckets = 0
    hash_key = None
    name = None

    input_dict = {
        "inputs": inputs,
        "num_buckets": num_buckets,
        "hash_key": hash_key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Three inputs
    st1 = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=np.array(['a', 'b'], dtype=np.string_), dense_shape=[2, 1])
    t2 = tf.constant([['c'], ['d']], dtype=tf.string)
    t3 = tf.constant([['e'], ['f']], dtype=tf.string)
    inputs = [st1, t2, t3]
    num_buckets = 0
    hash_key = None
    name = None

    input_dict = {
        "inputs": inputs,
        "num_buckets": num_buckets,
        "hash_key": hash_key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: different shapes
    st1 = tf.SparseTensor(indices=[[0, 0], [0, 1]], values=np.array(['a', 'b'], dtype=np.string_), dense_shape=[1, 2])
    t2 = tf.constant([['c'], ['d']], dtype=tf.string)
    inputs = [st1, t2]
    num_buckets = 0
    hash_key = None
    name = None

    input_dict = {
        "inputs": inputs,
        "num_buckets": num_buckets,
        "hash_key": hash_key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: num_buckets > 0 and hash_key provided
    st1 = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=np.array(['a', 'b'], dtype=np.string_), dense_shape=[2, 1])
    t2 = tf.constant([['c'], ['d']], dtype=tf.string)
    inputs = [st1, t2]
    num_buckets = 5
    hash_key = 123
    name = None

    input_dict = {
        "inputs": inputs,
        "num_buckets": num_buckets,
        "hash_key": hash_key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: name
    st1 = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=np.array(['a', 'b'], dtype=np.string_), dense_shape=[2, 1])
    st2 = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=np.array(['c', 'd'], dtype=np.string_), dense_shape=[2, 1])
    inputs = [st1, st2]
    num_buckets = 0
    hash_key = None
    name = "my_sparse_cross"

    input_dict = {
        "inputs": inputs,
        "num_buckets": num_buckets,
        "hash_key": hash_key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  SparseTensor with explicit empty indices
    st1 = tf.SparseTensor(indices=np.array([], dtype=np.int64).reshape(0,2), values=np.array([], dtype=np.string_), dense_shape=[2, 1])
    t2 = tf.constant([['c'], ['d']], dtype=tf.string)
    inputs = [st1, t2]
    num_buckets = 0
    hash_key = None
    name = None

    input_dict = {
        "inputs": inputs,
        "num_buckets": num_buckets,
        "hash_key": hash_key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: More Complex Shapes
    st1 = tf.SparseTensor(indices=[[0, 0, 0], [0, 1, 0]], values=np.array(['a', 'b'], dtype=np.string_), dense_shape=[1, 2, 1])
    t2 = tf.constant([[['c']], [['d']]], dtype=tf.string)
    inputs = [st1, t2]
    num_buckets = 0
    hash_key = None
    name = None

    input_dict = {
        "inputs": inputs,
        "num_buckets": num_buckets,
        "hash_key": hash_key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.cross_hashed"] = tf_sparse_cross_hashed_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.cross_hashed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.cross_hashed'.")

check_valid('tf.sparse.cross_hashed', generated_inputs['tf.sparse.cross_hashed'], lib="tf", suffix=0)
