
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
    name = "embedding_lookup_1"
    allow_fast_lookup = False
    input_dict = {'params': [params], 'sp_ids': sp_ids, 'sp_weights': sp_weights, 'combiner': combiner, 'max_norm': max_norm, 'name': name, 'allow_fast_lookup': allow_fast_lookup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    params = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0, 1, 2], dense_shape=(2, 2))
    sp_weights = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0.5, 1.0, 1.5], dense_shape=(2, 2))
    combiner = "mean"
    max_norm = 2.0
    name = "embedding_lookup_2"
    allow_fast_lookup = True
    input_dict = {'params': [params], 'sp_ids': sp_ids, 'sp_weights': sp_weights, 'combiner': combiner, 'max_norm': max_norm, 'name': name, 'allow_fast_lookup': allow_fast_lookup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    params = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [1, 0], [1, 1]], values=[0, 1, 2], dense_shape=(2, 2))
    sp_weights = tf.SparseTensor(indices=[[0, 0], [1, 0], [1, 1]], values=[1.0, 2.0, 0.5], dense_shape=(2, 2))
    combiner = "sqrtn"
    max_norm = 3.0
    name = "embedding_lookup_3"
    allow_fast_lookup = False
    input_dict = {'params': [params], 'sp_ids': sp_ids, 'sp_weights': sp_weights, 'combiner': combiner, 'max_norm': max_norm, 'name': name, 'allow_fast_lookup': allow_fast_lookup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    params = tf.constant([[1, 2], [3, 4], [5, 6], [7,8]], dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=[0, 3], dense_shape=(2, 1))
    sp_weights = None
    combiner = "sum"
    max_norm = None
    name = "embedding_lookup_4"
    allow_fast_lookup = False
    input_dict = {'params': [params], 'sp_ids': sp_ids, 'sp_weights': sp_weights, 'combiner': combiner, 'max_norm': max_norm, 'name': name, 'allow_fast_lookup': allow_fast_lookup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple Params

    params_0 = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    params_1 = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
    params = [params_0, params_1]


    sp_ids = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=[0, 1], dense_shape=(2, 1))
    sp_weights = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=[1.0, 1.0], dense_shape=(2, 1))
    combiner = "mean"
    max_norm = None
    name = "embedding_lookup_5"
    allow_fast_lookup = False
    input_dict = {'params': params, 'sp_ids': sp_ids, 'sp_weights': sp_weights, 'combiner': combiner, 'max_norm': max_norm, 'name': name, 'allow_fast_lookup': allow_fast_lookup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: RaggedTensor
    params = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    sp_ids = tf.ragged.constant([[0, 1], [2]])
    sp_weights = tf.ragged.constant([[0.5, 1.0], [1.5]])
    combiner = "sqrtn"
    max_norm = 2.5
    name = "embedding_lookup_6"
    allow_fast_lookup = True
    input_dict = {'params': [params], 'sp_ids': sp_ids, 'sp_weights': sp_weights, 'combiner': combiner, 'max_norm': max_norm, 'name': name, 'allow_fast_lookup': allow_fast_lookup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: RaggedTensor, no weights
    params = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    sp_ids = tf.ragged.constant([[0, 1], [2]])
    sp_weights = None
    combiner = "sum"
    max_norm = None
    name = "embedding_lookup_7"
    allow_fast_lookup = False
    input_dict = {'params': [params], 'sp_ids': sp_ids, 'sp_weights': sp_weights, 'combiner': combiner, 'max_norm': max_norm, 'name': name, 'allow_fast_lookup': allow_fast_lookup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different dense shape
    params = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [1, 0], [1, 1]], values=[0, 1, 2], dense_shape=(2, 2))
    sp_weights = tf.SparseTensor(indices=[[0, 0], [1, 0], [1, 1]], values=[1.0, 2.0, 0.5], dense_shape=(2, 2))
    combiner = "mean"
    max_norm = 3.0
    name = "embedding_lookup_8"
    allow_fast_lookup = False
    input_dict = {'params': [params], 'sp_ids': sp_ids, 'sp_weights': sp_weights, 'combiner': combiner, 'max_norm': max_norm, 'name': name, 'allow_fast_lookup': allow_fast_lookup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single id
    params = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0]], values=[0], dense_shape=(1, 1))
    sp_weights = tf.SparseTensor(indices=[[0, 0]], values=[1.0], dense_shape=(1, 1))
    combiner = "sum"
    max_norm = None
    name = "embedding_lookup_9"
    allow_fast_lookup = False
    input_dict = {'params': [params], 'sp_ids': sp_ids, 'sp_weights': sp_weights, 'combiner': combiner, 'max_norm': max_norm, 'name': name, 'allow_fast_lookup': allow_fast_lookup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multiple params, ragged ids, ragged weights
    params_0 = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    params_1 = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
    params = [params_0, params_1]
    sp_ids = tf.ragged.constant([[0, 1], [0]])
    sp_weights = tf.ragged.constant([[0.5, 1.0], [1.5]])
    combiner = "mean"
    max_norm = 2.5
    name = "embedding_lookup_10"
    allow_fast_lookup = True
    input_dict = {'params': params, 'sp_ids': sp_ids, 'sp_weights': sp_weights, 'combiner': combiner, 'max_norm': max_norm, 'name': name, 'allow_fast_lookup': allow_fast_lookup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.embedding_lookup_sparse"] = tf_nn_embedding_lookup_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.embedding_lookup_sparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.embedding_lookup_sparse'.")

check_valid('tf.nn.embedding_lookup_sparse', generated_inputs['tf.nn.embedding_lookup_sparse'], lib="tf", suffix=0)
