
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_embedding_lookup_sparse_inputs():
    """
    Generates a list of valid inputs for the tf.nn.embedding_lookup_sparse function.
    """
    list_of_inputs = []

    # To address the "'list' object has no attribute 'shape'" error, 'params'
    # is provided as a single numpy array instead of a list of arrays. This
    # corresponds to the non-sharded use case of the API.
    # To address the "'SparseTensor' object has no attribute 'size'" and the
    # "inputs should be in numpy format" constraints, sp_ids and sp_weights
    # are provided as RaggedTensors, which are a valid input type for the API
    # and might be better handled by the testing harness than SparseTensors.

    # Input 1: Basic case with combiner='sum'
    params_1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    sp_ids_1 = tf.ragged.constant([[0, 1], [3], [2]], dtype=tf.int64)
    
    input_dict_1 = {
        'params': [params_1],
        'sp_ids': sp_ids_1,
        'sp_weights': None,
        'combiner': 'sum',
        'max_norm': None,
        'name': 'test_1',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    
    # Input 2: With sp_weights
    params_2 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    sp_ids_2 = tf.ragged.constant([[0, 1], [3], [2]], dtype=tf.int64)
    sp_weights_2 = tf.ragged.constant([[0.1, 1.0], [0.5], [2.0]], dtype=tf.float32)

    input_dict_2 = {
        'params': [params_2],
        'sp_ids': sp_ids_2,
        'sp_weights': sp_weights_2,
        'combiner': 'sum',
        'max_norm': None,
        'name': 'test_2',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: combiner='mean'
    params_3 = np.random.rand(10, 5).astype(np.float32)
    sp_ids_3 = tf.ragged.constant([[1, 2], [3], [4, 5, 6]], dtype=tf.int64)
    sp_weights_3 = tf.ragged.constant([[0.2, 0.8], [1.0], [0.5, 0.3, 0.2]], dtype=tf.float32)
    
    input_dict_3 = {
        'params': [params_3],
        'sp_ids': sp_ids_3,
        'sp_weights': sp_weights_3,
        'combiner': 'mean',
        'max_norm': None,
        'name': 'test_3',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: combiner='sqrtn'
    input_dict_4 = copy.deepcopy(input_dict_3)
    input_dict_4['params'] = [np.random.rand(10, 5).astype(np.float32)]
    input_dict_4['combiner'] = 'sqrtn'
    input_dict_4['name'] = 'test_4'
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: With max_norm
    params_5 = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
    sp_ids_5 = tf.ragged.constant([[0, 1], [2]], dtype=tf.int64)

    input_dict_5 = {
        'params': [params_5],
        'sp_ids': sp_ids_5,
        'sp_weights': None,
        'combiner': 'sum',
        'max_norm': 35.0,
        'name': 'test_5',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Higher-dimensional params
    params_6 = np.random.rand(6, 2, 3).astype(np.float32)
    sp_ids_6 = tf.ragged.constant([[0, 5], [2]], dtype=tf.int64)
    
    input_dict_6 = {
        'params': [params_6],
        'sp_ids': sp_ids_6,
        'sp_weights': None,
        'combiner': 'mean',
        'max_norm': None,
        'name': 'test_6',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: allow_fast_lookup=True
    params_7 = np.random.rand(8, 5).astype(np.float32)
    sp_ids_7 = tf.ragged.constant([[1], [2, 3], [4]], dtype=tf.int64)

    input_dict_7 = {
        'params': [params_7],
        'sp_ids': sp_ids_7,
        'sp_weights': None,
        'combiner': 'sum',
        'max_norm': None,
        'name': 'test_7',
        'allow_fast_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: float64 dtype
    params_8 = np.random.rand(4, 2).astype(np.float64)
    sp_ids_8 = tf.ragged.constant([[0, 1], [2]], dtype=tf.int64)
    sp_weights_8 = tf.ragged.constant([[0.5, 0.5], [1.0]], dtype=tf.float64)

    input_dict_8 = {
        'params': [params_8],
        'sp_ids': sp_ids_8,
        'sp_weights': sp_weights_8,
        'combiner': 'mean',
        'max_norm': None,
        'name': 'test_8',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Sharded params
    params_9 = [np.random.rand(5, 4).astype(np.float32), np.random.rand(5, 4).astype(np.float32)]
    sp_ids_9 = tf.ragged.constant([[0, 6], [8], [3]], dtype=tf.int64)
    
    input_dict_9 = {
        'params': params_9,
        'sp_ids': sp_ids_9,
        'sp_weights': None,
        'combiner': 'mean',
        'max_norm': None,
        'name': 'test_9',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

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
