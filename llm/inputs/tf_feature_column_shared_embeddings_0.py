
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_shared_embeddings_inputs():
    list_of_inputs = []

    # Helper function to create categorical columns
    def create_categorical_column(key, vocabulary_list):
        return tf.feature_column.categorical_column_with_vocabulary_list(
            key=key, vocabulary_list=vocabulary_list, dtype=tf.string
        )

    # Input 1: Basic valid input
    categorical_columns = [
        create_categorical_column("col1", ["a", "b", "c"]),
        create_categorical_column("col2", ["a", "b", "c"]),
    ]
    dimension = 8
    input_dict = {
        "categorical_columns": categorical_columns,
        "dimension": dimension,
        "combiner": "mean",
        "initializer": tf.compat.v1.random_normal_initializer(mean=0.0, stddev=1.0),
        "shared_embedding_collection_name": "shared_embedding",
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": None,
        "trainable": True,
        "use_safe_embedding_lookup": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different dimension, combiner, and initializer
    categorical_columns = [
        create_categorical_column("col1", ["x", "y", "z"]),
        create_categorical_column("col2", ["x", "y", "z"]),
    ]
    dimension = 16
    input_dict = {
        "categorical_columns": categorical_columns,
        "dimension": dimension,
        "combiner": "sqrtn",
        "initializer": tf.compat.v1.zeros_initializer(),
        "shared_embedding_collection_name": "another_embedding",
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": 1.0,
        "trainable": False,
        "use_safe_embedding_lookup": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With max_norm
    categorical_columns = [
        create_categorical_column("col1", ["p", "q", "r"]),
        create_categorical_column("col2", ["p", "q", "r"]),
    ]
    dimension = 4
    input_dict = {
        "categorical_columns": categorical_columns,
        "dimension": dimension,
        "combiner": "sum",
        "initializer": tf.compat.v1.constant_initializer(0.5),
        "shared_embedding_collection_name": "embedding_sum",
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": 0.5,
        "trainable": True,
        "use_safe_embedding_lookup": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty collection name
    categorical_columns = [
        create_categorical_column("col1", ["1", "2", "3"]),
        create_categorical_column("col2", ["1", "2", "3"]),
    ]
    dimension = 32
    input_dict = {
        "categorical_columns": categorical_columns,
        "dimension": dimension,
        "combiner": "mean",
        "initializer": tf.compat.v1.random_uniform_initializer(minval=-1.0, maxval=1.0),
        "shared_embedding_collection_name": "",
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": None,
        "trainable": False,
        "use_safe_embedding_lookup": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different vocabulary
    categorical_columns = [
        create_categorical_column("col_a", ["alpha", "beta"]),
        create_categorical_column("col_b", ["alpha", "beta"]),
    ]
    dimension = 5
    input_dict = {
        "categorical_columns": categorical_columns,
        "dimension": dimension,
        "combiner": "sqrtn",
        "initializer": tf.compat.v1.variance_scaling_initializer(),
        "shared_embedding_collection_name": "vocab_diff",
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": 2.0,
        "trainable": True,
        "use_safe_embedding_lookup": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different initializer
    categorical_columns = [
        create_categorical_column("col_x", ["l", "m", "n"]),
        create_categorical_column("col_y", ["l", "m", "n"]),
    ]
    dimension = 7
    input_dict = {
        "categorical_columns": categorical_columns,
        "dimension": dimension,
        "combiner": "sum",
        "initializer": tf.compat.v1.glorot_normal_initializer(),
        "shared_embedding_collection_name": "glorot_init",
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": None,
        "trainable": False,
        "use_safe_embedding_lookup": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small dimension
    categorical_columns = [
        create_categorical_column("col_1", ["a", "b", "c"]),
        create_categorical_column("col_2", ["a", "b", "c"]),
    ]
    dimension = 1
    input_dict = {
        "categorical_columns": categorical_columns,
        "dimension": dimension,
        "combiner": "mean",
        "initializer": tf.compat.v1.random_normal_initializer(mean=0.0, stddev=0.1),
        "shared_embedding_collection_name": "dimension_1",
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": None,
        "trainable": True,
        "use_safe_embedding_lookup": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: ckpt_to_load_from and tensor_name_in_ckpt specified
    categorical_columns = [
        create_categorical_column("col1", ["a", "b", "c"]),
        create_categorical_column("col2", ["a", "b", "c"]),
    ]
    dimension = 8
    input_dict = {
        "categorical_columns": categorical_columns,
        "dimension": dimension,
        "combiner": "mean",
        "initializer": tf.compat.v1.random_normal_initializer(mean=0.0, stddev=1.0),
        "shared_embedding_collection_name": "shared_embedding",
        "ckpt_to_load_from": "model.ckpt",
        "tensor_name_in_ckpt": "embedding_weights",
        "max_norm": None,
        "trainable": True,
        "use_safe_embedding_lookup": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.shared_embeddings"] = tf_feature_column_shared_embeddings_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.shared_embeddings' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.shared_embeddings'.")

check_valid('tf.feature_column.shared_embeddings', generated_inputs['tf.feature_column.shared_embeddings'], lib="tf", suffix=0)
