
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_embedding_lookup_sparse_inputs():
    list_of_inputs = []

    # Input 1
    params = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [1, 0], [2, 0]], values=[0, 1, 2], dense_shape=(3, 1))
    sp_weights = tf.SparseTensor(indices=[[0, 0], [1, 0], [2, 0]], values=[1.0, 1.0, 1.0], dense_shape=(3, 1))
    combiner = "sum"
    max_norm = None
    name = None
    allow_fast_lookup = False

    input_dict = {
        "params": [params],
        "sp_ids": sp_ids,
        "sp_weights": sp_weights,
        "combiner": combiner,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    params = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]], values=[0, 1, 3, 2], dense_shape=(3, 2))
    sp_weights = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]], values=[0.1, 1.0, 0.5, 2.0], dense_shape=(3, 2))
    combiner = "mean"
    max_norm = 1.0
    name = "embedding_lookup"
    allow_fast_lookup = True

    input_dict = {
        "params": [params],
        "sp_ids": sp_ids,
        "sp_weights": sp_weights,
        "combiner": combiner,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    params_list = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32), tf.constant([[5, 6], [7, 8]], dtype=tf.float32)]
    sp_ids = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=[0, 1], dense_shape=(2, 1))
    sp_weights = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=[1.0, 1.0], dense_shape=(2, 1))
    combiner = "sqrtn"
    max_norm = None
    name = None
    allow_fast_lookup = False

    input_dict = {
        "params": params_list,
        "sp_ids": sp_ids,
        "sp_weights": sp_weights,
        "combiner": combiner,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: RaggedTensor sp_ids
    params = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=tf.float32)
    sp_ids = tf.ragged.constant([[0, 1], [3], [2]], dtype=tf.int64)
    sp_weights = tf.ragged.constant([[0.1, 1.0], [0.5], [2.0]], dtype=tf.float32)
    combiner = "sum"
    max_norm = None
    name = None
    allow_fast_lookup = False

    input_dict = {
        "params": [params],
        "sp_ids": sp_ids,
        "sp_weights": sp_weights,
        "combiner": combiner,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: RaggedTensor sp_ids, no weights
    params = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=tf.float32)
    sp_ids = tf.ragged.constant([[0, 1], [3], [2]], dtype=tf.int64)
    sp_weights = None
    combiner = "mean"
    max_norm = None
    name = None
    allow_fast_lookup = False

    input_dict = {
        "params": [params],
        "sp_ids": sp_ids,
        "sp_weights": sp_weights,
        "combiner": combiner,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: No weights, sum combiner
    params = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [1, 0], [2, 0]], values=[0, 1, 2], dense_shape=(3, 1))
    sp_weights = None
    combiner = "sum"
    max_norm = None
    name = None
    allow_fast_lookup = False

    input_dict = {
        "params": [params],
        "sp_ids": sp_ids,
        "sp_weights": sp_weights,
        "combiner": combiner,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Different sparse tensor shape
    params = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0, 1, 2], dense_shape=(2, 2))
    sp_weights = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1.0, 0.5, 1.5], dense_shape=(2, 2))
    combiner = "mean"
    max_norm = None
    name = None
    allow_fast_lookup = False

    input_dict = {
        "params": [params],
        "sp_ids": sp_ids,
        "sp_weights": sp_weights,
        "combiner": combiner,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different sparse tensor shape, no weights
    params = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0, 1, 2], dense_shape=(2, 2))
    sp_weights = None
    combiner = "sum"
    max_norm = None
    name = None
    allow_fast_lookup = False

    input_dict = {
        "params": [params],
        "sp_ids": sp_ids,
        "sp_weights": sp_weights,
        "combiner": combiner,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single value sparse tensor
    params = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0]], values=[0], dense_shape=(1, 1))
    sp_weights = tf.SparseTensor(indices=[[0, 0]], values=[1.0], dense_shape=(1, 1))
    combiner = "sum"
    max_norm = None
    name = None
    allow_fast_lookup = False

    input_dict = {
        "params": [params],
        "sp_ids": sp_ids,
        "sp_weights": sp_weights,
        "combiner": combiner,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: params as list with more elements, ensure indices are in range
    params_list = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32), tf.constant([[5, 6], [7, 8]], dtype=tf.float32), tf.constant([[9, 10], [11, 12]], dtype=tf.float32)]
    sp_ids = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=[0, 1], dense_shape=(2, 1))
    sp_weights = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=[1.0, 1.0], dense_shape=(2, 1))
    combiner = "sqrtn"
    max_norm = None
    name = None
    allow_fast_lookup = False

    input_dict = {
        "params": params_list,
        "sp_ids": sp_ids,
        "sp_weights": sp_weights,
        "combiner": combiner,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.embedding_lookup_sparse"] = tf_nn_embedding_lookup_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.embedding_lookup_sparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.embedding_lookup_sparse'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.embedding_lookup_sparse', generated_inputs['tf.nn.embedding_lookup_sparse'], lib="tf", suffix=0)
