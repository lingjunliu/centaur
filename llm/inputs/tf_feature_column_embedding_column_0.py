
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_embedding_column_inputs():
    list_of_inputs = []

    # Input 1
    categorical_column = [tf.feature_column.categorical_column_with_identity(key='test_column', num_buckets=10)]
    dimension = 8
    combiner = 'mean'
    initializer = tf.keras.initializers.truncated_normal(mean=0.0, stddev=0.1)
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = None
    trainable = True
    use_safe_embedding_lookup = True

    input_dict = {
        "categorical_column": categorical_column,
        "dimension": dimension,
        "combiner": combiner,
        "initializer": initializer,
        "ckpt_to_load_from": ckpt_to_load_from,
        "tensor_name_in_ckpt": tensor_name_in_ckpt,
        "max_norm": max_norm,
        "trainable": trainable,
        "use_safe_embedding_lookup": use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_list(key='test_column', vocabulary_list=['a', 'b', 'c'])]
    dimension = 16
    combiner = 'sqrtn'
    initializer = tf.keras.initializers.glorot_normal()
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 1.0
    trainable = False
    use_safe_embedding_lookup = False

    input_dict = {
        "categorical_column": categorical_column,
        "dimension": dimension,
        "combiner": combiner,
        "initializer": initializer,
        "ckpt_to_load_from": ckpt_to_load_from,
        "tensor_name_in_ckpt": tensor_name_in_ckpt,
        "max_norm": max_norm,
        "trainable": trainable,
        "use_safe_embedding_lookup": use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(key='test_column', hash_bucket_size=1000)]
    dimension = 32
    combiner = 'sum'
    initializer = tf.keras.initializers.zeros()
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 2.0
    trainable = True
    use_safe_embedding_lookup = True

    input_dict = {
        "categorical_column": categorical_column,
        "dimension": dimension,
        "combiner": combiner,
        "initializer": initializer,
        "ckpt_to_load_from": ckpt_to_load_from,
        "tensor_name_in_ckpt": tensor_name_in_ckpt,
        "max_norm": max_norm,
        "trainable": trainable,
        "use_safe_embedding_lookup": use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    categorical_column = [tf.feature_column.categorical_column_with_identity(key='test_column', num_buckets=50)]
    dimension = 4
    combiner = 'mean'
    initializer = tf.keras.initializers.ones()
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = None
    trainable = True
    use_safe_embedding_lookup = False

    input_dict = {
        "categorical_column": categorical_column,
        "dimension": dimension,
        "combiner": combiner,
        "initializer": initializer,
        "ckpt_to_load_from": ckpt_to_load_from,
        "tensor_name_in_ckpt": tensor_name_in_ckpt,
        "max_norm": max_norm,
        "trainable": trainable,
        "use_safe_embedding_lookup": use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_file(key='test_column', vocabulary_file='categories.txt', vocabulary_size=100)]
    dimension = 64
    combiner = 'sqrtn'
    initializer = tf.keras.initializers.random_uniform(minval=-0.05, maxval=0.05)
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 0.5
    trainable = False
    use_safe_embedding_lookup = True

    input_dict = {
        "categorical_column": categorical_column,
        "dimension": dimension,
        "combiner": combiner,
        "initializer": initializer,
        "ckpt_to_load_from": ckpt_to_load_from,
        "tensor_name_in_ckpt": tensor_name_in_ckpt,
        "max_norm": max_norm,
        "trainable": trainable,
        "use_safe_embedding_lookup": use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    categorical_column = [tf.feature_column.categorical_column_with_identity(key='column_a', num_buckets=20)]
    dimension = 128
    combiner = 'sum'
    initializer = tf.keras.initializers.he_normal()
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 5.0
    trainable = True
    use_safe_embedding_lookup = False

    input_dict = {
        "categorical_column": categorical_column,
        "dimension": dimension,
        "combiner": combiner,
        "initializer": initializer,
        "ckpt_to_load_from": ckpt_to_load_from,
        "tensor_name_in_ckpt": tensor_name_in_ckpt,
        "max_norm": max_norm,
        "trainable": trainable,
        "use_safe_embedding_lookup": use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(key='feature_name', hash_bucket_size=500)]
    dimension = 5
    combiner = 'mean'
    initializer = tf.keras.initializers.lecun_normal()
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 0.1
    trainable = True
    use_safe_embedding_lookup = True

    input_dict = {
        "categorical_column": categorical_column,
        "dimension": dimension,
        "combiner": combiner,
        "initializer": initializer,
        "ckpt_to_load_from": ckpt_to_load_from,
        "tensor_name_in_ckpt": tensor_name_in_ckpt,
        "max_norm": max_norm,
        "trainable": trainable,
        "use_safe_embedding_lookup": use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_list(key='city', vocabulary_list=['london', 'paris', 'tokyo'])]
    dimension = 7
    combiner = 'sqrtn'
    initializer = tf.keras.initializers.variance_scaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 0.75
    trainable = False
    use_safe_embedding_lookup = False

    input_dict = {
        "categorical_column": categorical_column,
        "dimension": dimension,
        "combiner": combiner,
        "initializer": initializer,
        "ckpt_to_load_from": ckpt_to_load_from,
        "tensor_name_in_ckpt": tensor_name_in_ckpt,
        "max_norm": max_norm,
        "trainable": trainable,
        "use_safe_embedding_lookup": use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    categorical_column = [tf.feature_column.categorical_column_with_identity(key='product_id', num_buckets=10000)]
    dimension = 3
    combiner = 'sum'
    initializer = tf.keras.initializers.orthogonal()
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 10.0
    trainable = True
    use_safe_embedding_lookup = True

    input_dict = {
        "categorical_column": categorical_column,
        "dimension": dimension,
        "combiner": combiner,
        "initializer": initializer,
        "ckpt_to_load_from": ckpt_to_load_from,
        "tensor_name_in_ckpt": tensor_name_in_ckpt,
        "max_norm": max_norm,
        "trainable": trainable,
        "use_safe_embedding_lookup": use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_file(key='item_category', vocabulary_file='categories.txt', vocabulary_size=500)]
    dimension = 1
    combiner = 'mean'
    initializer = tf.keras.initializers.identity()
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = None
    trainable = False
    use_safe_embedding_lookup = False

    input_dict = {
        "categorical_column": categorical_column,
        "dimension": dimension,
        "combiner": combiner,
        "initializer": initializer,
        "ckpt_to_load_from": ckpt_to_load_from,
        "tensor_name_in_ckpt": tensor_name_in_ckpt,
        "max_norm": max_norm,
        "trainable": trainable,
        "use_safe_embedding_lookup": use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11. Remove vocabulary_file. Also None tensor_name_in_ckpt
    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_list(key='color', vocabulary_list=['red', 'green', 'blue'])]
    dimension = 2
    combiner = 'mean'
    initializer = tf.keras.initializers.zeros()
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = None
    trainable = True
    use_safe_embedding_lookup = True

    input_dict = {
        "categorical_column": categorical_column,
        "dimension": dimension,
        "combiner": combiner,
        "initializer": initializer,
        "ckpt_to_load_from": ckpt_to_load_from,
        "tensor_name_in_ckpt": tensor_name_in_ckpt,
        "max_norm": max_norm,
        "trainable": trainable,
        "use_safe_embedding_lookup": use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 Different initializer
    categorical_column = [tf.feature_column.categorical_column_with_identity(key='test_column', num_buckets=20)]
    dimension = 10
    combiner = 'mean'
    initializer = tf.keras.initializers.random_normal(mean=0.0, stddev=0.05)
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = None
    trainable = True
    use_safe_embedding_lookup = False

    input_dict = {
        "categorical_column": categorical_column,
        "dimension": dimension,
        "combiner": combiner,
        "initializer": initializer,
        "ckpt_to_load_from": ckpt_to_load_from,
        "tensor_name_in_ckpt": tensor_name_in_ckpt,
        "max_norm": max_norm,
        "trainable": trainable,
        "use_safe_embedding_lookup": use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13 Different initializer and hash bucket size
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(key='test_column', hash_bucket_size=2000)]
    dimension = 40
    combiner = 'sum'
    initializer = tf.keras.initializers.glorot_uniform()
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 1.5
    trainable = True
    use_safe_embedding_lookup = True
    
    input_dict = {
        "categorical_column": categorical_column,
        "dimension": dimension,
        "combiner": combiner,
        "initializer": initializer,
        "ckpt_to_load_from": ckpt_to_load_from,
        "tensor_name_in_ckpt": tensor_name_in_ckpt,
        "max_norm": max_norm,
        "trainable": trainable,
        "use_safe_embedding_lookup": use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.embedding_column"] = tf_feature_column_embedding_column_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.embedding_column' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.embedding_column'.")

check_valid('tf.feature_column.embedding_column', generated_inputs['tf.feature_column.embedding_column'], lib="tf", suffix=0)
