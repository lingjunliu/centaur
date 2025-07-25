
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

if tf.__version__.startswith('2.'):
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

def tf_feature_column_shared_embeddings_inputs():
    """
    Generates a list of valid inputs for the tf.feature_column.shared_embeddings function.
    To avoid issues with static analysis tools that cannot compare tf.feature_column objects,
    the list of columns is wrapped in a numpy array with dtype=object.
    """
    list_of_inputs = []

    def create_initializer_tensor(vocab_size, dimension):
        return np.random.randn(vocab_size, dimension).astype(np.float32)

    # Input 1
    cols1 = np.array([
        tf.feature_column.categorical_column_with_identity('key1_1', num_buckets=100),
        tf.feature_column.categorical_column_with_identity('key2_1', num_buckets=100)
    ], dtype=object)
    list_of_inputs.append({
        'categorical_columns': cols1,
        'dimension': 8,
        'combiner': 'mean',
        'initializer': create_initializer_tensor(100, 8),
        'shared_embedding_collection_name': 'collection_1',
        'ckpt_to_load_from': '',
        'tensor_name_in_ckpt': '',
        'max_norm': 1.0,
        'trainable': True,
        'use_safe_embedding_lookup': True
    })

    # Input 2
    cols2 = np.array([
        tf.feature_column.categorical_column_with_hash_bucket('key1_2', hash_bucket_size=500),
        tf.feature_column.categorical_column_with_hash_bucket('key2_2', hash_bucket_size=500)
    ], dtype=object)
    list_of_inputs.append({
        'categorical_columns': cols2,
        'dimension': 16,
        'combiner': 'sqrtn',
        'initializer': create_initializer_tensor(500, 16),
        'shared_embedding_collection_name': 'collection_2',
        'ckpt_to_load_from': '',
        'tensor_name_in_ckpt': '',
        'max_norm': 5.0,
        'trainable': False,
        'use_safe_embedding_lookup': True
    })

    # Input 3
    cols3 = np.array([
        tf.feature_column.categorical_column_with_identity('key1_3', num_buckets=1000),
        tf.feature_column.categorical_column_with_identity('key2_3', num_buckets=1000)
    ], dtype=object)
    list_of_inputs.append({
        'categorical_columns': cols3,
        'dimension': 4,
        'combiner': 'sum',
        'initializer': create_initializer_tensor(1000, 4),
        'shared_embedding_collection_name': 'collection_3',
        'ckpt_to_load_from': '',
        'tensor_name_in_ckpt': '',
        'max_norm': 2.0,
        'trainable': True,
        'use_safe_embedding_lookup': False
    })
    
    # Input 4: Minimal dimension
    cols4 = np.array([
        tf.feature_column.categorical_column_with_identity('key1_4', num_buckets=20),
        tf.feature_column.categorical_column_with_identity('key2_4', num_buckets=20)
    ], dtype=object)
    list_of_inputs.append({
        'categorical_columns': cols4,
        'dimension': 1,
        'combiner': 'mean',
        'initializer': create_initializer_tensor(20, 1),
        'shared_embedding_collection_name': 'collection_4',
        'ckpt_to_load_from': '',
        'tensor_name_in_ckpt': '',
        'max_norm': 0.5,
        'trainable': True,
        'use_safe_embedding_lookup': True
    })

    # Input 5: Large dimension
    cols5 = np.array([
        tf.feature_column.categorical_column_with_hash_bucket('key1_5', hash_bucket_size=10000),
        tf.feature_column.categorical_column_with_hash_bucket('key2_5', hash_bucket_size=10000)
    ], dtype=object)
    list_of_inputs.append({
        'categorical_columns': cols5,
        'dimension': 128,
        'combiner': 'sqrtn',
        'initializer': create_initializer_tensor(10000, 128),
        'shared_embedding_collection_name': 'collection_5',
        'ckpt_to_load_from': '',
        'tensor_name_in_ckpt': '',
        'max_norm': 50.0,
        'trainable': True,
        'use_safe_embedding_lookup': True
    })

    # Input 6: Three shared columns
    cols6 = np.array([
        tf.feature_column.categorical_column_with_identity('key1_6', num_buckets=50),
        tf.feature_column.categorical_column_with_identity('key2_6', num_buckets=50),
        tf.feature_column.categorical_column_with_identity('key3_6', num_buckets=50)
    ], dtype=object)
    list_of_inputs.append({
        'categorical_columns': cols6,
        'dimension': 10,
        'combiner': 'sum',
        'initializer': create_initializer_tensor(50, 10),
        'shared_embedding_collection_name': 'collection_6',
        'ckpt_to_load_from': '',
        'tensor_name_in_ckpt': '',
        'max_norm': 3.0,
        'trainable': False,
        'use_safe_embedding_lookup': False
    })

    # Input 7: Loading from checkpoint
    cols7 = np.array([
        tf.feature_column.categorical_column_with_identity('key1_7', num_buckets=2000),
        tf.feature_column.categorical_column_with_identity('key2_7', num_buckets=2000)
    ], dtype=object)
    list_of_inputs.append({
        'categorical_columns': cols7,
        'dimension': 32,
        'combiner': 'mean',
        'initializer': create_initializer_tensor(2000, 32),
        'shared_embedding_collection_name': 'collection_7',
        'ckpt_to_load_from': './dummy_ckpt/model.ckpt',
        'tensor_name_in_ckpt': 'embedding_tensor_7',
        'max_norm': 4.0,
        'trainable': False,
        'use_safe_embedding_lookup': True
    })
    
    # Input 8
    cols8 = np.array([
        tf.feature_column.categorical_column_with_hash_bucket('key1_8', hash_bucket_size=300),
        tf.feature_column.categorical_column_with_hash_bucket('key2_8', hash_bucket_size=300)
    ], dtype=object)
    list_of_inputs.append({
        'categorical_columns': cols8,
        'dimension': 64,
        'combiner': 'sqrtn',
        'initializer': create_initializer_tensor(300, 64),
        'shared_embedding_collection_name': 'collection_8',
        'ckpt_to_load_from': '',
        'tensor_name_in_ckpt': '',
        'max_norm': 8.0,
        'trainable': True,
        'use_safe_embedding_lookup': False
    })
    
    # Input 9
    cols9 = np.array([
        tf.feature_column.categorical_column_with_identity('key1_9', num_buckets=150),
        tf.feature_column.categorical_column_with_identity('key2_9', num_buckets=150)
    ], dtype=object)
    list_of_inputs.append({
        'categorical_columns': cols9,
        'dimension': 12,
        'combiner': 'mean',
        'initializer': create_initializer_tensor(150, 12),
        'shared_embedding_collection_name': 'collection_9',
        'ckpt_to_load_from': '',
        'tensor_name_in_ckpt': '',
        'max_norm': 1.2,
        'trainable': True,
        'use_safe_embedding_lookup': True
    })

    # Input 10
    cols10 = np.array([
        tf.feature_column.categorical_column_with_hash_bucket('key1_10', hash_bucket_size=1234),
        tf.feature_column.categorical_column_with_hash_bucket('key2_10', hash_bucket_size=1234)
    ], dtype=object)
    list_of_inputs.append({
        'categorical_columns': cols10,
        'dimension': 20,
        'combiner': 'sum',
        'initializer': create_initializer_tensor(1234, 20),
        'shared_embedding_collection_name': 'collection_10',
        'ckpt_to_load_from': '',
        'tensor_name_in_ckpt': '',
        'max_norm': 2.5,
        'trainable': False,
        'use_safe_embedding_lookup': True
    })

    return list_of_inputs

generated_inputs["tf.feature_column.shared_embeddings"] = tf_feature_column_shared_embeddings_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.feature_column.shared_embeddings' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.shared_embeddings'.")

check_valid('tf.feature_column.shared_embeddings', generated_inputs['tf.feature_column.shared_embeddings'], lib="tf", suffix=0)
