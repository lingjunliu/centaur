
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_embedding_lookup_sparse_inputs():
    list_of_inputs = []

    # Input 1: Basic case with sum combiner
    params = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=[0, 1], dense_shape=(2, 1))
    sp_weights = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=[1.0, 2.0], dense_shape=(2, 1))
    input_dict = {
        'params': [params],
        'sp_ids': sp_ids,
        'sp_weights': sp_weights,
        'combiner': 'sum',
        'max_norm': None,
        'name': 'embedding_lookup_sum',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Mean combiner
    params = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0, 1, 2], dense_shape=(2, 2))
    sp_weights = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1.0, 1.0, 1.0], dense_shape=(2, 2))
    input_dict = {
        'params': [params],
        'sp_ids': sp_ids,
        'sp_weights': sp_weights,
        'combiner': 'mean',
        'max_norm': None,
        'name': 'embedding_lookup_mean',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Sqrtn combiner
    params = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0, 1, 2], dense_shape=(2, 2))
    sp_weights = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1.0, 1.0, 1.0], dense_shape=(2, 2))
    input_dict = {
        'params': [params],
        'sp_ids': sp_ids,
        'sp_weights': sp_weights,
        'combiner': 'sqrtn',
        'max_norm': None,
        'name': 'embedding_lookup_sqrtn',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With max_norm
    params = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0, 1, 2], dense_shape=(2, 2))
    sp_weights = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1.0, 1.0, 1.0], dense_shape=(2, 2))
    input_dict = {
        'params': [params],
        'sp_ids': sp_ids,
        'sp_weights': sp_weights,
        'combiner': 'sum',
        'max_norm': 3.0,
        'name': 'embedding_lookup_maxnorm',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: No weights
    params = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0, 1, 2], dense_shape=(2, 2))
    input_dict = {
        'params': [params],
        'sp_ids': sp_ids,
        'sp_weights': None,
        'combiner': 'sum',
        'max_norm': None,
        'name': 'embedding_lookup_noweights',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multiple params
    params1 = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    params2 = tf.constant([[7, 8], [9, 10], [11, 12]], dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[0, 1, 2], dense_shape=(2, 2))
    sp_weights = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1.0, 1.0, 1.0], dense_shape=(2, 2))
    input_dict = {
        'params': [params1, params2],
        'sp_ids': sp_ids,
        'sp_weights': sp_weights,
        'combiner': 'sum',
        'max_norm': None,
        'name': 'embedding_lookup_multipleparams',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different shape for sp_ids
    params = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [1, 0], [1, 1], [2, 0]], values=[0, 1, 2, 0], dense_shape=(3, 2))
    sp_weights = tf.SparseTensor(indices=[[0, 0], [1, 0], [1, 1], [2, 0]], values=[1.0, 1.0, 1.0, 1.0], dense_shape=(3, 2))
    input_dict = {
        'params': [params],
        'sp_ids': sp_ids,
        'sp_weights': sp_weights,
        'combiner': 'sum',
        'max_norm': None,
        'name': 'embedding_lookup_diffshape',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: allow_fast_lookup = True
    params = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=[0, 1], dense_shape=(2, 1))
    sp_weights = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=[1.0, 2.0], dense_shape=(2, 1))
    input_dict = {
        'params': [params],
        'sp_ids': sp_ids,
        'sp_weights': sp_weights,
        'combiner': 'sum',
        'max_norm': None,
        'name': 'embedding_lookup_sum',
        'allow_fast_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Ragged Tensor for sp_ids. Use tf.int64 for sp_ids values to avoid errors.
    params = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    sp_ids_ragged = tf.ragged.constant([[0, 1], [2]], dtype=tf.int64)
    sp_ids = sp_ids_ragged.to_sparse()
    sp_weights = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1.0, 2.0, 1.0], dense_shape=(2, 2))
    input_dict = {
        'params': [params],
        'sp_ids': sp_ids,
        'sp_weights': sp_weights,
        'combiner': 'sum',
        'max_norm': None,
        'name': 'embedding_lookup_ragged',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Ragged Tensor for sp_weights. Use tf.int64 for sp_ids values to avoid errors.
    params = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    sp_ids_ragged = tf.ragged.constant([[0, 1], [2]], dtype=tf.int64)
    sp_ids = sp_ids_ragged.to_sparse()
    sp_weights_ragged = tf.ragged.constant([[1.0, 2.0], [1.0]])
    sp_weights = sp_weights_ragged.to_sparse()

    input_dict = {
        'params': [params],
        'sp_ids': sp_ids,
        'sp_weights': sp_weights,
        'combiner': 'sum',
        'max_norm': None,
        'name': 'embedding_lookup_ragged_weights',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: params as a numpy array, ensure it is converted to tensor. Changed params name so it won't cause issues later.
    params_np = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    params_tensor = tf.convert_to_tensor(params_np, dtype=tf.float32)
    sp_ids = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=[0, 1], dense_shape=(2, 1))
    sp_weights = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=[1.0, 2.0], dense_shape=(2, 1))
    input_dict = {
        'params': [params_tensor],
        'sp_ids': sp_ids,
        'sp_weights': sp_weights,
        'combiner': 'sum',
        'max_norm': None,
        'name': 'embedding_lookup_sum',
        'allow_fast_lookup': False
    }
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
