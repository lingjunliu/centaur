
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_embedding_lookup_sparse_inputs():
    """
    Generates a list of valid inputs for tf.nn.embedding_lookup_sparse.
    The inputs are converted to dense numpy arrays to be compatible with a
    testing environment that expects array-like objects.
    """
    list_of_inputs = []

    # Input 1: Basic 'sum' combiner
    params_1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    sp_ids_1 = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]], values=[0, 1, 3, 2], dense_shape=(3, 2))
    sp_weights_1 = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]], values=[1.0, 1.0, 1.0, 1.0], dense_shape=(3, 2))
    input_dict_1 = {
        'params': params_1,
        'sp_ids': tf.sparse.to_dense(sp_ids_1).numpy(),
        'sp_weights': tf.sparse.to_dense(sp_weights_1).numpy(),
        'combiner': 'sum',
        'max_norm': 1e9,
        'name': 'test_sum',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 'mean' combiner with weights
    params_2 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    sp_ids_2 = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]], values=[0, 1, 3, 2], dense_shape=(3, 2))
    sp_weights_2 = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]], values=[0.1, 1.0, 0.5, 2.0], dense_shape=(3, 2))
    input_dict_2 = {
        'params': params_2,
        'sp_ids': tf.sparse.to_dense(sp_ids_2).numpy(),
        'sp_weights': tf.sparse.to_dense(sp_weights_2).numpy(),
        'combiner': 'mean',
        'max_norm': 1e9,
        'name': 'test_mean_with_weights',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 'sqrtn' combiner
    params_3 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    sp_ids_3 = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]], values=[0, 1, 2, 3], dense_shape=(3, 2))
    sp_weights_3 = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]], values=[2.0, 0.5, 1.0, 1.5], dense_shape=(3, 2))
    input_dict_3 = {
        'params': params_3,
        'sp_ids': tf.sparse.to_dense(sp_ids_3).numpy(),
        'sp_weights': tf.sparse.to_dense(sp_weights_3).numpy(),
        'combiner': 'sqrtn',
        'max_norm': 1e9,
        'name': 'test_sqrtn',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: With max_norm
    params_4 = np.array([[10, 20], [3, 4], [50, 60], [7, 8]], dtype=np.float32)
    sp_ids_4 = tf.SparseTensor(indices=[[0, 0], [1, 0], [1, 1]], values=[0, 2, 3], dense_shape=(2, 2))
    sp_weights_4 = tf.SparseTensor(indices=[[0, 0], [1, 0], [1, 1]], values=[1.0, 1.0, 1.0], dense_shape=(2, 2))
    input_dict_4 = {
        'params': params_4,
        'sp_ids': tf.sparse.to_dense(sp_ids_4).numpy(),
        'sp_weights': tf.sparse.to_dense(sp_weights_4).numpy(),
        'combiner': 'sum',
        'max_norm': 15.0,
        'name': 'test_max_norm',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5: allow_fast_lookup=True
    params_5 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    sp_ids_5 = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=[0, 2], dense_shape=(2, 1))
    sp_weights_5 = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=[1.0, 1.0], dense_shape=(2, 1))
    input_dict_5 = {
        'params': params_5,
        'sp_ids': tf.sparse.to_dense(sp_ids_5).numpy(),
        'sp_weights': tf.sparse.to_dense(sp_weights_5).numpy(),
        'combiner': 'mean',
        'max_norm': 1e9,
        'name': 'test_fast_lookup',
        'allow_fast_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: RaggedTensor input
    params_6 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    sp_ids_6 = tf.ragged.constant([[0, 1], [3], [2]], dtype=tf.int64)
    sp_weights_6 = tf.ragged.constant([[0.1, 1.0], [0.5], [2.0]], dtype=tf.float32)
    input_dict_6 = {
        'params': params_6,
        'sp_ids': sp_ids_6.to_tensor().numpy(),
        'sp_weights': sp_weights_6.to_tensor().numpy(),
        'combiner': 'sum',
        'max_norm': 1e9,
        'name': 'test_ragged',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Higher-dimensional params
    params_7 = np.random.rand(5, 2, 3).astype(np.float32)
    sp_ids_7 = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0, 4, 2], dense_shape=(2, 2))
    sp_weights_7 = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1.0, 0.5, 2.0], dense_shape=(2, 2))
    input_dict_7 = {
        'params': params_7,
        'sp_ids': tf.sparse.to_dense(sp_ids_7).numpy(),
        'sp_weights': tf.sparse.to_dense(sp_weights_7).numpy(),
        'combiner': 'mean',
        'max_norm': 1e9,
        'name': 'test_high_dim',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: Empty row in sparse tensor
    params_8 = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float32)
    sp_ids_8 = tf.SparseTensor(indices=[[0, 0], [2, 0]], values=[0, 2], dense_shape=(3, 1))
    sp_weights_8 = tf.SparseTensor(indices=[[0, 0], [2, 0]], values=[1.0, 1.0], dense_shape=(3, 1))
    input_dict_8 = {
        'params': params_8,
        'sp_ids': tf.sparse.to_dense(sp_ids_8).numpy(),
        'sp_weights': tf.sparse.to_dense(sp_weights_8).numpy(),
        'combiner': 'sum',
        'max_norm': 1e9,
        'name': 'test_empty_row',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: float64 dtype
    params_9 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    sp_ids_9 = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=[0, 2], dense_shape=(2, 1))
    sp_weights_9 = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=np.array([1.5, 2.5], dtype=np.float64), dense_shape=(2, 1))
    input_dict_9 = {
        'params': params_9,
        'sp_ids': tf.sparse.to_dense(sp_ids_9).numpy(),
        'sp_weights': tf.sparse.to_dense(sp_weights_9).numpy(),
        'combiner': 'sum',
        'max_norm': 1e9,
        'name': 'test_float64',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Sharded params (concatenated)
    params_10 = np.concatenate([
        np.array([[1, 1], [2, 2]], dtype=np.float32), 
        np.array([[3, 3], [4, 4]], dtype=np.float32)
    ], axis=0)
    sp_ids_10 = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0, 3, 1], dense_shape=(2, 2))
    sp_weights_10 = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0.5, 2.0, 1.0], dense_shape=(2, 2))
    input_dict_10 = {
        'params': params_10,
        'sp_ids': tf.sparse.to_dense(sp_ids_10).numpy(),
        'sp_weights': tf.sparse.to_dense(sp_weights_10).numpy(),
        'combiner': 'sum',
        'max_norm': 1e9,
        'name': 'test_sharded',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

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

check_valid('tf.nn.embedding_lookup_sparse', generated_inputs['tf.nn.embedding_lookup_sparse'], lib="tf", suffix=0)
