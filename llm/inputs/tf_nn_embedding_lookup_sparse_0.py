
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# This is the key change. By enabling this, tf.Tensors will behave like
# numpy arrays and have attributes like .size, which should resolve the
# harness's AttributeError on EagerTensor objects.
tf.experimental.numpy.experimental_enable_numpy_behavior()

def tf_nn_embedding_lookup_sparse_inputs():
    """
    Generates a list of valid inputs for tf.nn.embedding_lookup_sparse.
    - Uses tf.experimental.numpy.experimental_enable_numpy_behavior() to make
      tf.Tensor objects compatible with the testing harness.
    - `params` is provided as a single tensor. While the API accepts a list of
      tensors, the testing harness has been observed to fail on list inputs.
      A single tensor is a valid input type for this parameter.
    - `sp_ids` and `sp_weights` are provided as `tf.SparseTensor` as required
      by the API, which resolves the TypeError during execution.
    """
    list_of_inputs = []

    # --- Common Input Components ---
    params_single = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=tf.float32)
    sp_ids_1 = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]],
                              values=tf.constant([0, 1, 3, 2], dtype=tf.int64),
                              dense_shape=(3, 2))
    sp_weights_1 = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]],
                                   values=tf.constant([0.1, 1.0, 0.5, 2.0], dtype=tf.float32),
                                   dense_shape=(3, 2))

    # --- Input Cases ---
    
    # Input 1: Basic case with combiner='sum', no weights.
    input_dict_1 = {
        'params': copy.deepcopy(params_single),
        'sp_ids': copy.deepcopy(sp_ids_1),
        'sp_weights': None,
        'combiner': 'sum',
        'max_norm': None,
        'name': 'test_1',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(input_dict_1)

    # Input 2: Basic case with weights and combiner='sum'.
    input_dict_2 = {
        'params': copy.deepcopy(params_single),
        'sp_ids': copy.deepcopy(sp_ids_1),
        'sp_weights': copy.deepcopy(sp_weights_1),
        'combiner': 'sum',
        'max_norm': None,
        'name': 'test_2',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(input_dict_2)

    # Input 3: Combiner='mean', no weights.
    input_dict_3 = {
        'params': copy.deepcopy(params_single),
        'sp_ids': copy.deepcopy(sp_ids_1),
        'sp_weights': None,
        'combiner': 'mean',
        'max_norm': None,
        'name': 'test_3',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(input_dict_3)

    # Input 4: Combiner='sqrtn', with weights.
    input_dict_4 = {
        'params': copy.deepcopy(params_single),
        'sp_ids': copy.deepcopy(sp_ids_1),
        'sp_weights': copy.deepcopy(sp_weights_1),
        'combiner': 'sqrtn',
        'max_norm': None,
        'name': 'test_4',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(input_dict_4)
    
    # Input 5: With max_norm.
    params_5 = tf.constant([[10., 20.], [30., 40.], [50., 60.]], dtype=tf.float32)
    sp_ids_5 = tf.SparseTensor(indices=[[0, 0], [0, 1]], values=tf.constant([0, 1], dtype=tf.int64), dense_shape=(1, 2))
    input_dict_5 = {
        'params': params_5,
        'sp_ids': sp_ids_5,
        'sp_weights': None,
        'combiner': 'sum',
        'max_norm': 15.0,
        'name': 'test_5',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(input_dict_5)

    # Input 6: params with more than 2 dimensions.
    params_6 = tf.constant([[[1, 2], [3, 4]],
                             [[5, 6], [7, 8]],
                             [[9, 10], [11, 12]],
                             [[13, 14], [15, 16]]], dtype=tf.float32)
    input_dict_6 = {
        'params': params_6,
        'sp_ids': copy.deepcopy(sp_ids_1),
        'sp_weights': None,
        'combiner': 'sum',
        'max_norm': None,
        'name': 'test_6',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(input_dict_6)
    
    # Input 7: float64 dtype.
    params_7 = tf.constant([[1.1, 2.2], [3.3, 4.4]], dtype=tf.float64)
    sp_ids_7 = tf.SparseTensor(indices=[[0, 0]], values=tf.constant([1], dtype=tf.int64), dense_shape=(1, 1))
    sp_weights_7 = tf.SparseTensor(indices=[[0, 0]], values=tf.constant([0.5], dtype=tf.float64), dense_shape=(1, 1))
    input_dict_7 = {
        'params': params_7,
        'sp_ids': sp_ids_7,
        'sp_weights': sp_weights_7,
        'combiner': 'mean',
        'max_norm': None,
        'name': 'test_7',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(input_dict_7)
    
    # Input 8: allow_fast_lookup=True.
    input_dict_8 = {
        'params': copy.deepcopy(params_single),
        'sp_ids': copy.deepcopy(sp_ids_1),
        'sp_weights': None,
        'combiner': 'sum',
        'max_norm': None,
        'name': 'test_8',
        'allow_fast_lookup': True
    }
    list_of_inputs.append(input_dict_8)
    
    # Input 9: Different sp_ids shape.
    params_9 = tf.constant([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], dtype=tf.float32)
    sp_ids_9 = tf.SparseTensor(indices=[[0, 0], [0, 2], [1, 1], [1, 3], [2, 0]],
                                   values=tf.constant([0, 1, 2, 3, 4], dtype=tf.int64),
                                   dense_shape=(3, 4))
    input_dict_9 = {
        'params': params_9,
        'sp_ids': sp_ids_9,
        'sp_weights': None,
        'combiner': 'mean',
        'max_norm': None,
        'name': 'test_9',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(input_dict_9)

    # Input 10: Combiner='sqrtn' without weights
    input_dict_10 = {
        'params': copy.deepcopy(params_single),
        'sp_ids': copy.deepcopy(sp_ids_1),
        'sp_weights': None,
        'combiner': 'sqrtn',
        'max_norm': None,
        'name': 'test_10',
        'allow_fast_lookup': False,
    }
    list_of_inputs.append(input_dict_10)

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
