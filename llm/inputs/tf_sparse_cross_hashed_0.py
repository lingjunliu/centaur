
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_cross_hashed_inputs():
    list_of_inputs = []

    def to_numpy(x):
        if isinstance(x, tf.SparseTensor):
            return tf.sparse.to_dense(x).numpy()
        elif hasattr(x, 'numpy'):
            return x.numpy()
        else:
            return x

    # Input 1
    inputs = [tf.sparse.from_dense(tf.constant([["a", "b"], ["c", "d"]])), tf.constant([["e"], ["f"]])]
    inputs = [to_numpy(x) for x in inputs]
    num_buckets = 10
    hash_key = 123
    name = "sparse_cross_1"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = [tf.sparse.from_dense(tf.constant([["a"]])), tf.constant([["b"]])]
    inputs = [to_numpy(x) for x in inputs]
    num_buckets = 0
    hash_key = None
    name = None
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = [tf.sparse.from_dense(tf.constant([["a", "b", "c"]])), tf.constant([["d", "e", "f"]])]
    inputs = [to_numpy(x) for x in inputs]
    num_buckets = 5
    hash_key = 456
    name = "sparse_cross_3"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = [tf.sparse.from_dense(tf.constant([["1", "2"], ["3", "4"]])), tf.constant([["5"], ["6"]])]
    inputs = [to_numpy(x) for x in inputs]
    num_buckets = 100
    hash_key = 789
    name = "sparse_cross_4"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    inputs = [tf.sparse.from_dense(tf.constant([["a"], ["b"]])), tf.constant([["c"], ["d"]]), tf.constant([["e"], ["f"]])]
    inputs = [to_numpy(x) for x in inputs]
    num_buckets = 2
    hash_key = 101
    name = "sparse_cross_5"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = [tf.sparse.from_dense(tf.constant([["a", "b"]])), tf.constant([["c"], ["d"]])]
    inputs = [to_numpy(x) for x in inputs]
    num_buckets = 1
    hash_key = 202
    name = "sparse_cross_6"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = [tf.sparse.from_dense(tf.constant([["A", "B", "C"], ["D", "E", "F"]])), tf.constant([["G"], ["H"]])]
    inputs = [to_numpy(x) for x in inputs]
    num_buckets = 15
    hash_key = 303
    name = "sparse_cross_7"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    inputs = [tf.sparse.from_dense(tf.constant([["x", "y"], ["z", "w"]])), tf.constant([["p"], ["q"]]), tf.sparse.from_dense(tf.constant([["r"], ["s"]]))]
    inputs = [to_numpy(x) for x in inputs]
    num_buckets = 7
    hash_key = 404
    name = "sparse_cross_8"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = [tf.sparse.from_dense(tf.constant([["1", "2", "3"]])), tf.constant([["4", "5", "6"]])]
    inputs = [to_numpy(x) for x in inputs]
    num_buckets = 20
    hash_key = 505
    name = "sparse_cross_9"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    inputs = [tf.sparse.from_dense(tf.constant([["val1", "val2"], ["val3", "val4"]])), tf.constant([["val5"], ["val6"]])]
    inputs = [to_numpy(x) for x in inputs]
    num_buckets = 3
    hash_key = 606
    name = "sparse_cross_10"
    input_dict = {"inputs": inputs, "num_buckets": num_buckets, "hash_key": hash_key, "name": name}
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
