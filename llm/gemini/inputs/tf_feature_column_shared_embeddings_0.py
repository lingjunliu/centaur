
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_feature_column_shared_embeddings_inputs():
    list_of_inputs = []

    def initializer_fn_1(shape, dtype=None, partition_info=None):
        return tf.random.normal(shape, dtype=dtype).numpy()

    def initializer_fn_2(shape, dtype=None, partition_info=None):
        return tf.zeros(shape, dtype=dtype).numpy()
    
    def initializer_fn_3(shape, dtype=None, partition_info=None):
        return tf.ones(shape, dtype=dtype).numpy()

    # Input 1
    categorical_column_1 = tf.feature_column.categorical_column_with_identity(key='video_id_1', num_buckets=1000)
    categorical_column_2 = tf.feature_column.categorical_column_with_identity(key='video_id_2', num_buckets=1000)
    categorical_columns = [categorical_column_1, categorical_column_2]
    dimension = 16
    combiner = 'mean'
    initializer = initializer_fn_1
    shared_embedding_collection_name = 'video_embeddings'
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

    return list_of_inputs

generated_inputs = {}
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


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.feature_column.shared_embeddings', generated_inputs['tf.feature_column.shared_embeddings'], lib="tf", suffix=0)
