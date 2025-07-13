
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

    # Input 1: Basic case
    categorical_columns1 = [create_categorical_column("col1", ["a", "b", "c"])]
    dimension1 = 8
    initializer1 = tf.random.uniform(shape=[8,3], minval=0.0, maxval=1.0)
    input_dict1 = {
        "categorical_columns": categorical_columns1,
        "dimension": dimension1,
        "combiner": "mean",
        "initializer": initializer1,
        "shared_embedding_collection_name": "shared_embedding",
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": None,
        "trainable": True,
        "use_safe_embedding_lookup": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

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
