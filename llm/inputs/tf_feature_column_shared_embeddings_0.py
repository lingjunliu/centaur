
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
            key=key, vocabulary_list=vocabulary_list)

    # Input 1
    categorical_columns = [
        create_categorical_column("video_id_1", ["a", "b", "c"]),
        create_categorical_column("video_id_2", ["a", "b", "c"])
    ]
    dimension = 8
    combiner = "mean"
    initializer = tf.compat.v1.truncated_normal_initializer(mean=0.0, stddev=0.1)
    shared_embedding_collection_name = "shared_video_embeddings"
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = None
    trainable = True
    use_safe_embedding_lookup = True

    input_dict = {
        "categorical_columns": categorical_columns,
        "dimension": dimension,
        "combiner": combiner,
        "initializer": initializer,
        "shared_embedding_collection_name": shared_embedding_collection_name,
        "ckpt_to_load_from": ckpt_to_load_from,
        "tensor_name_in_ckpt": tensor_name_in_ckpt,
        "max_norm": max_norm,
        "trainable": trainable,
        "use_safe_embedding_lookup": use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: No shared_embedding_collection_name, other parameters varied
    categorical_columns = [
        create_categorical_column("platform_1", ["web", "mobile", "desktop"]),
        create_categorical_column("platform_2", ["web", "mobile", "desktop"])
    ]
    dimension = 9
    combiner = "sqrtn"
    initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.05)
    shared_embedding_collection_name = None
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 0.75
    trainable = True
    use_safe_embedding_lookup = False

    input_dict = {
        "categorical_columns": categorical_columns,
        "dimension": dimension,
        "combiner": combiner,
        "initializer": initializer,
        "shared_embedding_collection_name": shared_embedding_collection_name,
        "ckpt_to_load_from": ckpt_to_load_from,
        "tensor_name_in_ckpt": tensor_name_in_ckpt,
        "max_norm": max_norm,
        "trainable": trainable,
        "use_safe_embedding_lookup": use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    #Input 11: Another different combination of arguments
    categorical_columns = [
        create_categorical_column("browser_1", ["chrome", "firefox", "safari"]),
        create_categorical_column("browser_2", ["chrome", "firefox", "safari"])
    ]
    dimension = 10
    combiner = "mean"
    initializer = tf.compat.v1.glorot_uniform_initializer()
    shared_embedding_collection_name = "browser_embedding"
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 0.9
    trainable = False
    use_safe_embedding_lookup = True

    input_dict = {
        "categorical_columns": categorical_columns,
        "dimension": dimension,
        "combiner": combiner,
        "initializer": initializer,
        "shared_embedding_collection_name": shared_embedding_collection_name,
        "ckpt_to_load_from": ckpt_to_load_from,
        "tensor_name_in_ckpt": tensor_name_in_ckpt,
        "max_norm": max_norm,
        "trainable": trainable,
        "use_safe_embedding_lookup": use_safe_embedding_lookup
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
