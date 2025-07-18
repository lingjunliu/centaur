
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_nn_safe_embedding_lookup_sparse_inputs():
    """
    Generates a list of valid inputs for the tf.nn.safe_embedding_lookup_sparse function.
    """
    list_of_inputs = []

    # Input 1: Basic case with a single embedding tensor and 'mean' combiner.
    input_dict_1 = {
        'embedding_weights': np.random.rand(10, 3).astype(np.float32),
        'sparse_ids': tf.SparseTensor(
            indices=tf.constant([[0, 0], [0, 1], [1, 0], [2, 0]], dtype=tf.int64),
            values=tf.constant([1, 2, 3, 4], dtype=tf.int64),
            dense_shape=tf.constant([3, 5], dtype=tf.int64)
        ),
        'sparse_weights': None,
        'combiner': 'mean',
        'default_id': None,
        'max_norm': None,
        'name': 'basic_mean',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: With explicit sparse_weights and 'sum' combiner.
    input_dict_2 = {
        'embedding_weights': np.random.rand(10, 4).astype(np.float32),
        'sparse_ids': tf.SparseTensor(
            indices=tf.constant([[0, 0], [0, 1], [2, 0], [3, 1]], dtype=tf.int64),
            values=tf.constant([0, 1, 2, 3], dtype=tf.int64),
            dense_shape=tf.constant([4, 2], dtype=tf.int64)
        ),
        'sparse_weights': tf.SparseTensor(
            indices=tf.constant([[0, 0], [0, 1], [2, 0], [3, 1]], dtype=tf.int64),
            values=tf.constant([1.0, 0.5, 2.0, 1.5], dtype=tf.float32),
            dense_shape=tf.constant([4, 2], dtype=tf.int64)
        ),
        'combiner': 'sum',
        'default_id': None,
        'max_norm': None,
        'name': 'with_weights_sum',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: With a default_id for empty feature rows.
    input_dict_3 = {
        'embedding_weights': np.random.rand(5, 2).astype(np.float32),
        'sparse_ids': tf.SparseTensor(
            indices=tf.constant([[0, 0], [2, 1], [2, 2]], dtype=tf.int64),
            values=tf.constant([1, 3, 4], dtype=tf.int64),
            dense_shape=tf.constant([4, 3], dtype=tf.int64)
        ),
        'sparse_weights': None,
        'combiner': 'mean',
        'default_id': 0,
        'max_norm': None,
        'name': 'with_default_id',
        'allow_fast_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: With max_norm for L2 normalization and 'sqrtn' combiner.
    input_dict_4 = {
        'embedding_weights': np.random.rand(8, 5).astype(np.float32),
        'sparse_ids': tf.SparseTensor(
            indices=tf.constant([[0, 0], [0, 1], [1, 0]], dtype=tf.int64),
            values=tf.constant([1, 7, 5], dtype=tf.int64),
            dense_shape=tf.constant([2, 2], dtype=tf.int64)
        ),
        'sparse_weights': None,
        'combiner': 'sqrtn',
        'default_id': None,
        'max_norm': 1.5,
        'name': 'with_max_norm_sqrtn',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: With invalid IDs (<0) and non-positive weights to be pruned.
    input_dict_5 = {
        'embedding_weights': np.random.rand(10, 2).astype(np.float32),
        'sparse_ids': tf.SparseTensor(
            indices=tf.constant([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]], dtype=tf.int64),
            values=tf.constant([1, -1, 3, 5, 9], dtype=tf.int64),
            dense_shape=tf.constant([2, 3], dtype=tf.int64)
        ),
        'sparse_weights': tf.SparseTensor(
            indices=tf.constant([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]], dtype=tf.int64),
            values=tf.constant([2.0, 1.0, 0.0, -0.5, 3.0], dtype=tf.float32),
            dense_shape=tf.constant([2, 3], dtype=tf.int64)
        ),
        'combiner': 'sum',
        'default_id': 1,
        'max_norm': None,
        'name': 'pruning_invalids',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: With 3D sparse_ids.
    input_dict_6 = {
        'embedding_weights': np.random.rand(20, 10).astype(np.float32),
        'sparse_ids': tf.SparseTensor(
            indices=tf.constant([[0, 0, 1], [0, 2, 3], [1, 1, 0], [1, 1, 2]], dtype=tf.int64),
            values=tf.constant([1, 2, 3, 4], dtype=tf.int64),
            dense_shape=tf.constant([2, 3, 5], dtype=tf.int64)
        ),
        'sparse_weights': None,
        'combiner': 'mean',
        'default_id': None,
        'max_norm': None,
        'name': '3d_sparse_ids',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: High-dimensional embeddings (embedding vectors are matrices).
    input_dict_7 = {
        'embedding_weights': np.random.rand(5, 2, 3).astype(np.float32),
        'sparse_ids': tf.SparseTensor(
            indices=tf.constant([[0, 1], [1, 2], [1, 3], [3, 0]], dtype=tf.int64),
            values=tf.constant([0, 1, 2, 3], dtype=tf.int64),
            dense_shape=tf.constant([4, 5], dtype=tf.int64)
        ),
        'sparse_weights': None,
        'combiner': 'sum',
        'default_id': None,
        'max_norm': None,
        'name': 'high_dim_embeddings',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Using float64 and int64 dtypes.
    input_dict_8 = {
        'embedding_weights': np.random.rand(10, 5).astype(np.float64),
        'sparse_ids': tf.SparseTensor(
            indices=tf.constant([[0, 0], [1, 1]], dtype=tf.int64),
            values=tf.constant([1, 2], dtype=tf.int64),
            dense_shape=tf.constant([2, 2], dtype=tf.int64)
        ),
        'sparse_weights': tf.SparseTensor(
            indices=tf.constant([[0, 0], [1, 1]], dtype=tf.int64),
            values=tf.constant([1.0, 2.0], dtype=np.float64),
            dense_shape=tf.constant([2, 2], dtype=tf.int64)
        ),
        'combiner': 'mean',
        'default_id': 0,
        'max_norm': 5.0,
        'name': 'float64_dtypes',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: A row with only invalid IDs, should return default_id embedding.
    input_dict_9 = {
        'embedding_weights': np.random.rand(5, 2).astype(np.float32),
        'sparse_ids': tf.SparseTensor(
            indices=tf.constant([[0, 0], [0, 1]], dtype=tf.int64),
            values=tf.constant([-1, -5], dtype=tf.int64),
            dense_shape=tf.constant([2, 2], dtype=tf.int64)
        ),
        'sparse_weights': None,
        'combiner': 'mean',
        'default_id': 2,
        'max_norm': None,
        'name': 'all_pruned_row',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Mix of empty, pruned, and valid rows in a single batch.
    input_dict_10 = {
        'embedding_weights': np.random.rand(10, 3).astype(np.float32),
        'sparse_ids': tf.SparseTensor(
            indices=tf.constant([[0, 0], [0, 1], [2, 0], [3, 0], [3, 1]], dtype=tf.int64),
            values=tf.constant([1, 2, -3, 4, -5], dtype=tf.int64),
            dense_shape=tf.constant([4, 2], dtype=tf.int64)
        ),
        'sparse_weights': None,
        'combiner': 'mean',
        'default_id': 0,
        'max_norm': None,
        'name': 'mixed_batch',
        'allow_fast_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.nn.safe_embedding_lookup_sparse"] = get_tf_nn_safe_embedding_lookup_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.safe_embedding_lookup_sparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.safe_embedding_lookup_sparse'.")

check_valid('tf.nn.safe_embedding_lookup_sparse', generated_inputs['tf.nn.safe_embedding_lookup_sparse'], lib="tf", suffix=0)
