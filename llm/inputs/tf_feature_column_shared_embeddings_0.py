
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# This function is deprecated and requires TF1 compatibility mode.
tf.compat.v1.disable_eager_execution()

def tf_feature_column_shared_embeddings_inputs():
    """
    Generates a list of valid inputs for tf.feature_column.shared_embeddings.
    """
    list_of_inputs = []

    def create_initializer(shape):
        return np.random.uniform(low=-1.0, high=1.0, size=shape).astype(np.float32)

    # All inputs will use the simplest categorical column, `categorical_column_with_identity`,
    # to avoid complex object structures that might confuse the testing framework.

    # Input 1: Basic case with 'mean' combiner
    cat_cols_1 = [
        tf.feature_column.categorical_column_with_identity(key='video_id_1', num_buckets=100),
        tf.feature_column.categorical_column_with_identity(key='impression_id_1', num_buckets=100)
    ]
    input_dict_1 = {
        'categorical_columns': cat_cols_1,
        'dimension': 8,
        'combiner': 'mean',
        'initializer': create_initializer((100, 8)),
        'shared_embedding_collection_name': 'collection_1',
        'ckpt_to_load_from': '',
        'tensor_name_in_ckpt': '',
        'max_norm': 1.0,
        'trainable': True,
        'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 'sqrtn' combiner, not trainable
    cat_cols_2 = [
        tf.feature_column.categorical_column_with_identity(key='feature_a_2', num_buckets=50),
        tf.feature_column.categorical_column_with_identity(key='feature_b_2', num_buckets=50)
    ]
    input_dict_2 = {
        'categorical_columns': cat_cols_2,
        'dimension': 16,
        'combiner': 'sqrtn',
        'initializer': create_initializer((50, 16)),
        'shared_embedding_collection_name': 'collection_2',
        'ckpt_to_load_from': '',
        'tensor_name_in_ckpt': '',
        'max_norm': 2.5,
        'trainable': False,
        'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 'sum' combiner, no safe lookup
    cat_cols_3 = [
        tf.feature_column.categorical_column_with_identity(key='product_id_3', num_buckets=200),
        tf.feature_column.categorical_column_with_identity(key='category_id_3', num_buckets=200)
    ]
    input_dict_3 = {
        'categorical_columns': cat_cols_3,
        'dimension': 32,
        'combiner': 'sum',
        'initializer': create_initializer((200, 32)),
        'shared_embedding_collection_name': 'collection_3',
        'ckpt_to_load_from': '',
        'tensor_name_in_ckpt': '',
        'max_norm': 0.5,
        'trainable': True,
        'use_safe_embedding_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: With checkpoint loading
    cat_cols_4 = [
        tf.feature_column.categorical_column_with_identity(key='user_id_4', num_buckets=1000),
        tf.feature_column.categorical_column_with_identity(key='author_id_4', num_buckets=1000)
    ]
    input_dict_4 = {
        'categorical_columns': cat_cols_4,
        'dimension': 64,
        'combiner': 'mean',
        'initializer': create_initializer((1000, 64)),
        'shared_embedding_collection_name': 'user_author_embeddings',
        'ckpt_to_load_from': '/tmp/model.ckpt',
        'tensor_name_in_ckpt': 'user_author_embeddings/embedding_weights',
        'max_norm': 10.0,
        'trainable': True,
        'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: More columns (3) and smaller dimension
    cat_cols_5 = [
        tf.feature_column.categorical_column_with_identity(key='tag_1_5', num_buckets=20),
        tf.feature_column.categorical_column_with_identity(key='tag_2_5', num_buckets=20),
        tf.feature_column.categorical_column_with_identity(key='tag_3_5', num_buckets=20)
    ]
    input_dict_5 = {
        'categorical_columns': cat_cols_5,
        'dimension': 2,
        'combiner': 'sqrtn',
        'initializer': create_initializer((20, 2)),
        'shared_embedding_collection_name': 'tag_embeddings',
        'ckpt_to_load_from': '',
        'tensor_name_in_ckpt': '',
        'max_norm': 1.0,
        'trainable': True,
        'use_safe_embedding_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Minimal dimension (1)
    cat_cols_6 = [
        tf.feature_column.categorical_column_with_identity(key='min_dim_1_6', num_buckets=10),
        tf.feature_column.categorical_column_with_identity(key='min_dim_2_6', num_buckets=10)
    ]
    input_dict_6 = {
        'categorical_columns': cat_cols_6,
        'dimension': 1,
        'combiner': 'mean',
        'initializer': create_initializer((10, 1)),
        'shared_embedding_collection_name': 'min_dim_collection',
        'ckpt_to_load_from': '',
        'tensor_name_in_ckpt': '',
        'max_norm': 1.0,
        'trainable': True,
        'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Zero initializer, large dimension
    cat_cols_7 = [
        tf.feature_column.categorical_column_with_identity(key='large_dim_1_7', num_buckets=500),
        tf.feature_column.categorical_column_with_identity(key='large_dim_2_7', num_buckets=500)
    ]
    input_dict_7 = {
        'categorical_columns': cat_cols_7,
        'dimension': 256,
        'combiner': 'sqrtn',
        'initializer': np.zeros((500, 256), dtype=np.float32),
        'shared_embedding_collection_name': 'large_dim_collection',
        'ckpt_to_load_from': '',
        'tensor_name_in_ckpt': '',
        'max_norm': 100.0,
        'trainable': True,
        'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Large num_buckets
    cat_cols_8 = [
        tf.feature_column.categorical_column_with_identity(key='big_vocab_1', num_buckets=10000),
        tf.feature_column.categorical_column_with_identity(key='big_vocab_2', num_buckets=10000)
    ]
    input_dict_8 = {
        'categorical_columns': cat_cols_8,
        'dimension': 128,
        'combiner': 'mean',
        'initializer': create_initializer((10000, 128)),
        'shared_embedding_collection_name': 'big_vocab_collection',
        'ckpt_to_load_from': '',
        'tensor_name_in_ckpt': '',
        'max_norm': 10.0,
        'trainable': True,
        'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Only one column in the list
    cat_cols_9 = [
        tf.feature_column.categorical_column_with_identity(key='single_col_9', num_buckets=5)
    ]
    input_dict_9 = {
        'categorical_columns': cat_cols_9,
        'dimension': 3,
        'combiner': 'mean',
        'initializer': create_initializer((5, 3)),
        'shared_embedding_collection_name': 'single_collection',
        'ckpt_to_load_from': '',
        'tensor_name_in_ckpt': '',
        'max_norm': 1.0,
        'trainable': True,
        'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Another simple case with different values
    cat_cols_10 = [
        tf.feature_column.categorical_column_with_identity(key='ad_id_10', num_buckets=5000),
        tf.feature_column.categorical_column_with_identity(key='campaign_id_10', num_buckets=5000)
    ]
    input_dict_10 = {
        'categorical_columns': cat_cols_10,
        'dimension': 50,
        'combiner': 'sum',
        'initializer': create_initializer((5000, 50)),
        'shared_embedding_collection_name': 'ad_collection',
        'ckpt_to_load_from': '',
        'tensor_name_in_ckpt': '',
        'max_norm': 5.0,
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
