
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os

def tf_feature_column_shared_embeddings_inputs():
    try:
        tf.compat.v1.disable_eager_execution()
    except (AttributeError, RuntimeError):
        pass

    # Setup a dummy checkpoint file required by some inputs
    ckpt_dir = "tf_shared_embeddings_ckpt_dir"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "model.ckpt")
    # Create dummy files to make the path valid for tf.train.latest_checkpoint
    with open(ckpt_path + ".index", "w") as f:
        f.write("dummy")
    with open(ckpt_path + ".data-00000-of-00001", "w") as f:
        f.write("dummy")

    list_of_inputs = []

    # All inputs will now have concrete values for optional parameters
    # to strictly adhere to the provided signature and avoid None where
    # a specific type like 'string' or 'float' is expected.

    # Input 1
    num_buckets_1, dim_1 = 10, 8
    cat_cols_1 = [tf.feature_column.categorical_column_with_identity('col1', num_buckets=num_buckets_1)]
    input_dict_1 = {
        'categorical_columns': cat_cols_1,
        'dimension': dim_1,
        'combiner': 'mean',
        'initializer': np.random.randn(num_buckets_1, dim_1).astype(np.float32),
        'shared_embedding_collection_name': 'collection_1',
        'ckpt_to_load_from': ckpt_path,
        'tensor_name_in_ckpt': 'tensor_1',
        'max_norm': 1.0,
        'trainable': True,
        'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2
    num_buckets_2, dim_2 = 100, 16
    cat_cols_2 = [
        tf.feature_column.categorical_column_with_identity('feat_a', num_buckets=num_buckets_2),
        tf.feature_column.categorical_column_with_identity('feat_b', num_buckets=num_buckets_2)
    ]
    input_dict_2 = {
        'categorical_columns': cat_cols_2,
        'dimension': dim_2,
        'combiner': 'sqrtn',
        'initializer': np.zeros((num_buckets_2, dim_2), dtype=np.float32),
        'shared_embedding_collection_name': 'collection_2',
        'ckpt_to_load_from': ckpt_path,
        'tensor_name_in_ckpt': 'tensor_2',
        'max_norm': 2.0,
        'trainable': False,
        'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3
    num_buckets_3, dim_3 = 50, 4
    cat_cols_3 = [tf.feature_column.categorical_column_with_identity('user_id', num_buckets=num_buckets_3)]
    input_dict_3 = {
        'categorical_columns': cat_cols_3,
        'dimension': dim_3,
        'combiner': 'sum',
        'initializer': np.ones((num_buckets_3, dim_3), dtype=np.float32),
        'shared_embedding_collection_name': 'user_embeddings',
        'ckpt_to_load_from': ckpt_path,
        'tensor_name_in_ckpt': 'user_tensor',
        'max_norm': 0.5,
        'trainable': True,
        'use_safe_embedding_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4
    num_buckets_4, dim_4 = 200, 32
    cat_cols_4 = [
        tf.feature_column.categorical_column_with_identity('prod_view', num_buckets=num_buckets_4),
        tf.feature_column.categorical_column_with_identity('prod_buy', num_buckets=num_buckets_4)
    ]
    input_dict_4 = {
        'categorical_columns': cat_cols_4,
        'dimension': dim_4,
        'combiner': 'mean',
        'initializer': np.random.uniform(-1., 1., size=(num_buckets_4, dim_4)).astype(np.float32),
        'shared_embedding_collection_name': 'prod_collection',
        'ckpt_to_load_from': ckpt_path,
        'tensor_name_in_ckpt': 'prod_tensor',
        'max_norm': 5.0,
        'trainable': True,
        'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5
    num_buckets_5, dim_5 = 5, 1
    cat_cols_5 = [tf.feature_column.categorical_column_with_identity('minimal_col', num_buckets=num_buckets_5)]
    input_dict_5 = {
        'categorical_columns': cat_cols_5,
        'dimension': dim_5,
        'combiner': 'mean',
        'initializer': np.random.randn(num_buckets_5, dim_5).astype(np.float32),
        'shared_embedding_collection_name': 'minimal_collection',
        'ckpt_to_load_from': ckpt_path,
        'tensor_name_in_ckpt': 'minimal_tensor',
        'max_norm': 1.5,
        'trainable': True,
        'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6
    num_buckets_6, dim_6 = 1000, 128
    cat_cols_6 = [
        tf.feature_column.categorical_column_with_identity('id_a', num_buckets=num_buckets_6),
        tf.feature_column.categorical_column_with_identity('id_b', num_buckets=num_buckets_6),
        tf.feature_column.categorical_column_with_identity('id_c', num_buckets=num_buckets_6),
    ]
    input_dict_6 = {
        'categorical_columns': cat_cols_6,
        'dimension': dim_6,
        'combiner': 'sqrtn',
        'initializer': np.random.randn(num_buckets_6, dim_6).astype(np.float32) * 0.1,
        'shared_embedding_collection_name': 'large_id_space',
        'ckpt_to_load_from': ckpt_path,
        'tensor_name_in_ckpt': 'large_tensor',
        'max_norm': 10.0,
        'trainable': True,
        'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7
    num_buckets_7, dim_7 = 42, 12
    cat_cols_7 = [
        tf.feature_column.categorical_column_with_identity('full_col1', num_buckets=num_buckets_7),
        tf.feature_column.categorical_column_with_identity('full_col2', num_buckets=num_buckets_7)
    ]
    input_dict_7 = {
        'categorical_columns': cat_cols_7,
        'dimension': dim_7,
        'combiner': 'sum',
        'initializer': np.random.uniform(-0.5, 0.5, size=(num_buckets_7, dim_7)).astype(np.float32),
        'shared_embedding_collection_name': 'full_spec_collection',
        'ckpt_to_load_from': ckpt_path,
        'tensor_name_in_ckpt': 'another_tensor_name',
        'max_norm': 2.5,
        'trainable': False,
        'use_safe_embedding_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8
    num_buckets_8, dim_8 = 3, 3
    cat_cols_8 = [tf.feature_column.categorical_column_with_identity('tiny_col', num_buckets=num_buckets_8)]
    input_dict_8 = {
        'categorical_columns': cat_cols_8,
        'dimension': dim_8,
        'combiner': 'sum',
        'initializer': np.ones((num_buckets_8, dim_8), dtype=np.float32),
        'shared_embedding_collection_name': 'tiny_collection',
        'ckpt_to_load_from': ckpt_path,
        'tensor_name_in_ckpt': 'tiny_tensor',
        'max_norm': 1.0,
        'trainable': False,
        'use_safe_embedding_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9
    num_buckets_9, dim_9 = 7, 7
    cat_cols_9 = [tf.feature_column.categorical_column_with_identity('another_col', num_buckets=num_buckets_9)]
    input_dict_9 = {
        'categorical_columns': cat_cols_9,
        'dimension': dim_9,
        'combiner': 'mean',
        'initializer': np.eye(num_buckets_9, M=dim_9, dtype=np.float32),
        'shared_embedding_collection_name': 'another_collection',
        'ckpt_to_load_from': ckpt_path,
        'tensor_name_in_ckpt': 'another_tensor',
        'max_norm': 100.0,
        'trainable': True,
        'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10
    num_buckets_10, dim_10 = 2, 64
    cat_cols_10 = [tf.feature_column.categorical_column_with_identity('binary_col', num_buckets=num_buckets_10)]
    input_dict_10 = {
        'categorical_columns': cat_cols_10,
        'dimension': dim_10,
        'combiner': 'sqrtn',
        'initializer': np.random.randn(num_buckets_10, dim_10).astype(np.float32),
        'shared_embedding_collection_name': 'binary_collection',
        'ckpt_to_load_from': ckpt_path,
        'tensor_name_in_ckpt': 'binary_tensor',
        'max_norm': 50.0,
        'trainable': False,
        'use_safe_embedding_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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
