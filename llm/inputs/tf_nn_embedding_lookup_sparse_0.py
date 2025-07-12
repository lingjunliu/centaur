
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_embedding_lookup_sparse_inputs():
    list_of_inputs = []

    # Input 1: Basic example with sum combiner
    params = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]], values=[0, 1, 3, 2], dense_shape=(3, 2))
    sp_weights = None
    combiner = "sum"
    max_norm = None
    name = "embedding_lookup_1"
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

    # Input 2: Example with sparse weights and mean combiner
    params = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]], values=[0, 1, 3, 2], dense_shape=(3, 2))
    sparse_weights = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]], values=[0.1, 1.0, 0.5, 2.0], dense_shape=(3, 2))
    combiner = "mean"
    max_norm = None
    name = "embedding_lookup_2"
    allow_fast_lookup = False

    input_dict = {
        "params": [params],
        "sp_ids": sp_ids,
        "sp_weights": sparse_weights,
        "combiner": combiner,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Example with max_norm clipping
    params = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]], values=[0, 1, 3, 2], dense_shape=(3, 2))
    sp_weights = None
    combiner = "sum"
    max_norm = 5.0
    name = "embedding_lookup_3"
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

    # Input 4: Example with sqrtn combiner
    params = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]], values=[0, 1, 3, 2], dense_shape=(3, 2))
    sparse_weights = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]], values=[0.1, 1.0, 0.5, 2.0], dense_shape=(3, 2))
    combiner = "sqrtn"
    max_norm = None
    name = "embedding_lookup_4"
    allow_fast_lookup = False

    input_dict = {
        "params": [params],
        "sp_ids": sp_ids,
        "sp_weights": sparse_weights,
        "combiner": combiner,
        "max_norm": max_norm,
        "name": name,
        "allow_fast_lookup": allow_fast_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Example with RaggedTensor
    params = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    sp_ids = tf.ragged.constant([[0, 1], [3], [2]])
    sp_weights = None
    combiner = "sum"
    max_norm = None
    name = "embedding_lookup_5"
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

    # Input 6: RaggedTensor with weights
    params = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    sp_ids = tf.ragged.constant([[0, 1], [3], [2]])
    sp_weights = tf.ragged.constant([[0.1, 1.0], [0.5], [2.0]])
    combiner = "mean"
    max_norm = None
    name = "embedding_lookup_6"
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

    # Input 7: allow_fast_lookup = True
    params = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]], values=[0, 1, 3, 2], dense_shape=(3, 2))
    sp_weights = None
    combiner = "sum"
    max_norm = None
    name = "embedding_lookup_7"
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
    
    # Input 8: Multiple params tensors
    params_list = [np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32), np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)]
    sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]], values=[0, 1, 1, 1], dense_shape=(3, 2))
    sp_weights = None
    combiner = "sum"
    max_norm = None
    name = "embedding_lookup_8"
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

    # Input 9: Different shape params tensors
    def to_numpy(x):
        if isinstance(x, tf.Tensor):
            return x.numpy()
        return x
    params_list = [np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]], dtype=np.float32), np.array([[7, 8, 9], [9, 10, 11], [11, 12, 13]], dtype=np.float32)]
    sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]], values=[0, 1, 1, 1], dense_shape=(3, 2))
    sp_weights = None
    combiner = "sum"
    max_norm = None
    name = "embedding_lookup_9"
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

    # Input 10: Single value sparse tensor.
    params = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    sp_ids = tf.SparseTensor(indices=[[0,0]], values=[0], dense_shape=(1, 1))
    sp_weights = None
    combiner = "sum"
    max_norm = None
    name = "embedding_lookup_10"
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
    
    # Input 11: params is a list of numpy arrays
    params_list = [np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32), np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)]
    sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0, 1, 0], dense_shape=(3, 2))
    sp_weights = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0.1, 1.0, 0.5], dense_shape=(3, 2))
    combiner = "mean"
    max_norm = 5.0
    name = "embedding_lookup_11"
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
    
    # Input 12: RaggedTensor with empty rows
    params = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    sp_ids = tf.ragged.constant([[0, 1], [], [2]])
    sp_weights = None
    combiner = "sum"
    max_norm = None
    name = "embedding_lookup_12"
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

    return list_of_inputs

generated_inputs["tf.nn.embedding_lookup_sparse"] = tf_nn_embedding_lookup_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.embedding_lookup_sparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.embedding_lookup_sparse'.")

check_valid('tf.nn.embedding_lookup_sparse', generated_inputs['tf.nn.embedding_lookup_sparse'], lib="tf", suffix=0)
