
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_safe_embedding_lookup_sparse_inputs():
    list_of_inputs = []

    # Input 1
    embedding_weights = [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)]
    sparse_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0, 1, 2], dense_shape=[2, 2])
    sparse_weights = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1.0, 2.0, 3.0], dense_shape=[2, 2])
    combiner = "mean"
    default_id = 0
    max_norm = None
    name = "embedding_lookup_1"
    allow_fast_lookup = False
    input_dict = {'embedding_weights': embedding_weights, 'sparse_ids': sparse_ids, 'sparse_weights': sparse_weights, 'combiner': combiner, 'default_id': default_id, 'max_norm': max_norm, 'name': name, 'allow_fast_lookup': allow_fast_lookup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    embedding_weights = [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)]
    sparse_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [1,1]], values=[0, 1, -1, 3], dense_shape=[2, 2])
    sparse_weights = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [1,1]], values=[1.0, 2.0, 3.0, 0.5], dense_shape=[2, 2])
    combiner = "sum"
    default_id = 1
    max_norm = 1.0
    name = "embedding_lookup_2"
    allow_fast_lookup = True
    input_dict = {'embedding_weights': embedding_weights, 'sparse_ids': sparse_ids, 'sparse_weights': sparse_weights, 'combiner': combiner, 'default_id': default_id, 'max_norm': max_norm, 'name': name, 'allow_fast_lookup': allow_fast_lookup}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    embedding_weights = [np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)]
    sparse_ids = tf.SparseTensor(indices=[[0, 0], [1, 1]], values=[0, 1], dense_shape=[2, 2])
    sparse_weights = None
    combiner = "mean"
    default_id = 0
    max_norm = None
    name = "embedding_lookup_4"
    allow_fast_lookup = False
    input_dict = {'embedding_weights': embedding_weights, 'sparse_ids': sparse_ids, 'sparse_weights': sparse_weights, 'combiner': combiner, 'default_id': default_id, 'max_norm': max_norm, 'name': name, 'allow_fast_lookup': allow_fast_lookup}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.safe_embedding_lookup_sparse"] = tf_nn_safe_embedding_lookup_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.safe_embedding_lookup_sparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.safe_embedding_lookup_sparse'.")

check_valid('tf.nn.safe_embedding_lookup_sparse', generated_inputs['tf.nn.safe_embedding_lookup_sparse'], lib="tf", suffix=0)
