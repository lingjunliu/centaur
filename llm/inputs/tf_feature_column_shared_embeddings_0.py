
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_shared_embeddings_inputs():
    list_of_inputs = []

    def create_categorical_column(key, vocabulary_list):
        return tf.feature_column.categorical_column_with_vocabulary_list(
            key=key, vocabulary_list=vocabulary_list, dtype=tf.string
        )

    # Input 1, valid
    categorical_columns = [
        create_categorical_column("col1", ["a", "b", "c"]),
        create_categorical_column("col2", ["a", "b", "c"])
    ]
    dimension = 8
    combiner = "mean"
    initializer = tf.compat.v1.random_normal_initializer(mean=0.0, stddev=0.1)
    shared_embedding_collection_name = "shared_embeddings_1"
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

    # Input 2, valid
    categorical_columns = [
        create_categorical_column("col1", ["x", "y", "z"]),
        create_categorical_column("col2", ["x", "y", "z"])
    ]
    dimension = 16
    combiner = "sqrtn"
    initializer = tf.compat.v1.zeros_initializer()
    shared_embedding_collection_name = "shared_embeddings_2"
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 1.0
    trainable = False
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

    # Input 3, valid
    categorical_columns = [
        create_categorical_column("col1", ["p", "q", "r"]),
        create_categorical_column("col2", ["p", "q", "r"])
    ]
    dimension = 4
    combiner = "sum"
    initializer = tf.compat.v1.ones_initializer()
    shared_embedding_collection_name = "shared_embeddings_3"
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 0.5
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

     # Input 4, valid
    categorical_columns = [
        create_categorical_column("col1", ["one", "two", "three"]),
        create_categorical_column("col2", ["one", "two", "three"])
    ]
    dimension = 32
    combiner = "mean"
    initializer = tf.compat.v1.glorot_uniform_initializer()
    shared_embedding_collection_name = "shared_embeddings_4"
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 2.0
    trainable = False
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

    # Input 5, valid
    categorical_columns = [
        create_categorical_column("col1", ["alpha", "beta", "gamma"]),
        create_categorical_column("col2", ["alpha", "beta", "gamma"])
    ]
    dimension = 1
    combiner = "sqrtn"
    initializer = tf.compat.v1.constant_initializer(value=0.5)
    shared_embedding_collection_name = "shared_embeddings_5"
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 0.25
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

    # Input 6, valid, different vocab
    categorical_columns = [
        create_categorical_column("col1", ["a", "b", "c", "d"]),
        create_categorical_column("col2", ["a", "b", "c", "d"])
    ]
    dimension = 10
    combiner = "mean"
    initializer = tf.compat.v1.random_uniform_initializer(minval=-0.1, maxval=0.1)
    shared_embedding_collection_name = "shared_embeddings_6"
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 1.5
    trainable = False
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

    # Input 7, valid
    categorical_columns = [
        create_categorical_column("col1", ["yes", "no", "maybe"]),
        create_categorical_column("col2", ["yes", "no", "maybe"])
    ]
    dimension = 7
    combiner = "sum"
    initializer = tf.compat.v1.truncated_normal_initializer(mean=0.0, stddev=0.05)
    shared_embedding_collection_name = "shared_embeddings_7"
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

    # Input 8, valid
    categorical_columns = [
        create_categorical_column("col1", ["high", "low", "medium"]),
        create_categorical_column("col2", ["high", "low", "medium"])
    ]
    dimension = 2
    combiner = "mean"
    initializer = tf.compat.v1.variance_scaling_initializer()
    shared_embedding_collection_name = "shared_embeddings_8"
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 0.75
    trainable = False
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

    # Input 9, valid
    categorical_columns = [
        create_categorical_column("col1", ["hot", "cold", "warm"]),
        create_categorical_column("col2", ["hot", "cold", "warm"])
    ]
    dimension = 5
    combiner = "sqrtn"
    initializer = tf.compat.v1.orthogonal_initializer()
    shared_embedding_collection_name = "shared_embeddings_9"
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 1.25
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

    # Input 10, valid, weighted categorical column
    def create_weighted_categorical_column(key, vocabulary_list):
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key=key, vocabulary_list=vocabulary_list, dtype=tf.string
        )
        return tf.feature_column.weighted_categorical_column(categorical_column=cat_col, weight_feature_key="weights")
    
    categorical_columns = [
        create_weighted_categorical_column("col1", ["A", "B", "C"]),
        create_weighted_categorical_column("col2", ["A", "B", "C"])
    ]
    dimension = 6
    combiner = "sum"
    initializer = tf.compat.v1.initializers.identity()
    shared_embedding_collection_name = "shared_embeddings_10"
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = None
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
    
    # Input 11, valid, initializer as a tensor
    categorical_columns = [
        create_categorical_column("col1", ["A", "B"]),
        create_categorical_column("col2", ["A", "B"])
    ]
    dimension = 3
    combiner = "mean"
    initializer = tf.constant([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=tf.float32).numpy()
    shared_embedding_collection_name = "shared_embeddings_11"
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 1.0
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
    
    # Input 12, valid, initializer as a numpy array
    categorical_columns = [
        create_categorical_column("col1", ["X", "Y"]),
        create_categorical_column("col2", ["X", "Y"])
    ]
    dimension = 4
    combiner = "sum"
    initializer = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
    shared_embedding_collection_name = "shared_embeddings_12"
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 0.5
    trainable = False
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
    
    # Input 13, valid, initializer as a numpy array with dimension=1
    categorical_columns = [
        create_categorical_column("col1", ["U", "V"]),
        create_categorical_column("col2", ["U", "V"])
    ]
    dimension = 1
    combiner = "mean"
    initializer = np.array([[0.1], [0.2]], dtype=np.float32)
    shared_embedding_collection_name = "shared_embeddings_13"
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 0.25
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
    
    # Input 14, valid, ckpt loading
    categorical_columns = [
        create_categorical_column("col1", ["A1", "B1"]),
        create_categorical_column("col2", ["A1", "B1"])
    ]
    dimension = 2
    combiner = "mean"
    initializer = tf.compat.v1.random_normal_initializer(mean=0.0, stddev=0.1)
    shared_embedding_collection_name = "shared_embeddings_14"
    ckpt_to_load_from = "model.ckpt"
    tensor_name_in_ckpt = "embedding_tensor"
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
    
    print("Valid")

if 'tf.feature_column.shared_embeddings' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.shared_embeddings'.")

check_valid('tf.feature_column.shared_embeddings', generated_inputs['tf.feature_column.shared_embeddings'], lib="tf", suffix=0)
