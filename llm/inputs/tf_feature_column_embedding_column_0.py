
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_embedding_column_inputs():
    list_of_inputs = []

    categorical_column_1 = tf.feature_column.categorical_column_with_identity(num_buckets=5, key='test_col1')
    categorical_column_2 = tf.feature_column.categorical_column_with_vocabulary_list(vocabulary_list=['a', 'b', 'c'], key='test_col2')
    categorical_column_3 = tf.feature_column.categorical_column_with_hash_bucket(hash_bucket_size=10, key='test_col3')

    def create_initializer(mean=0.0, stddev=0.1):
        def initializer(shape, dtype=tf.float32):
            return tf.compat.v1.truncated_normal(shape, mean=mean, stddev=stddev, dtype=dtype)
        return initializer

    # Input 1, valid
    input_dict = {
        "categorical_column": categorical_column_1,
        "dimension": 8,
        "combiner": 'mean',
        "initializer": create_initializer(),
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": None,
        "trainable": True,
        "use_safe_embedding_lookup": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def create_initializer_zeros():
        def initializer(shape, dtype=tf.float32):
            return tf.zeros(shape, dtype=dtype)
        return initializer

    # Input 2, valid
    input_dict = {
        "categorical_column": categorical_column_2,
        "dimension": 16,
        "combiner": 'sqrtn',
        "initializer": create_initializer_zeros(),
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": 1.0,
        "trainable": False,
        "use_safe_embedding_lookup": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def create_initializer_glorot():
        def initializer(shape, dtype=tf.float32):
            limit = np.sqrt(6.0 / (shape[0] + shape[1])) if len(shape) > 1 else np.sqrt(6.0 / shape[0])
            return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype)
        return initializer

    # Input 3, valid
    input_dict = {
        "categorical_column": categorical_column_3,
        "dimension": 4,
        "combiner": 'sum',
        "initializer": create_initializer_glorot(),
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": 0.5,
        "trainable": True,
        "use_safe_embedding_lookup": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def create_initializer_uniform(minval=-0.5, maxval=0.5):
        def initializer(shape, dtype=tf.float32):
            return tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
        return initializer

    # Input 4, valid
    input_dict = {
        "categorical_column": categorical_column_1,
        "dimension": 32,
        "combiner": 'mean',
        "initializer": create_initializer_uniform(),
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": None,
        "trainable": True,
        "use_safe_embedding_lookup": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def create_initializer_ones():
        def initializer(shape, dtype=tf.float32):
            return tf.ones(shape, dtype=dtype)
        return initializer

   # Input 5, valid
    input_dict = {
        "categorical_column": categorical_column_2,
        "dimension": 128,
        "combiner": 'sqrtn',
        "initializer": create_initializer_ones(),
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": 2.0,
        "trainable": False,
        "use_safe_embedding_lookup": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def create_initializer_constant(value=0.1):
        def initializer(shape, dtype=tf.float32):
            return tf.constant(value=value, dtype=dtype, shape=shape)
        return initializer

    # Input 6, valid
    input_dict = {
        "categorical_column": categorical_column_3,
        "dimension": 2,
        "combiner": 'sum',
        "initializer": create_initializer_constant(),
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": 1.5,
        "trainable": True,
        "use_safe_embedding_lookup": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def create_initializer_variance():
        def initializer(shape, dtype=tf.float32):
            scale = 1.0
            fan_in, fan_out = shape[0], shape[1] if len(shape) > 1 else shape[0]
            limit = np.sqrt(scale * 2.0 / (fan_in + fan_out))
            return tf.random.truncated_normal(shape, 0.0, limit / np.sqrt(3.0), dtype=dtype)

        return initializer

    # Input 7, valid
    input_dict = {
       "categorical_column": categorical_column_1,
        "dimension": 64,
        "combiner": 'mean',
        "initializer": create_initializer_variance(),
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": None,
        "trainable": True,
        "use_safe_embedding_lookup": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def create_initializer_orthogonal():
        def initializer(shape, dtype=tf.float32):
            flat_shape = (shape[0], np.prod(shape[1:])) if len(shape) > 1 else (shape[0], 1)
            a = np.random.normal(0.0, 1.0, flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == flat_shape else v
            q = q.reshape(shape)
            return tf.constant(q, dtype=dtype, shape=shape)

        return initializer

    # Input 8, valid
    input_dict = {
        "categorical_column": categorical_column_2,
        "dimension": 1,
        "combiner": 'sqrtn',
        "initializer": create_initializer_orthogonal(),
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": 0.75,
        "trainable": False,
        "use_safe_embedding_lookup": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def create_initializer_glorot_normal():
        def initializer(shape, dtype=tf.float32):
            stddev = np.sqrt(2.0 / (shape[0] + shape[1])) if len(shape) > 1 else np.sqrt(2.0 / shape[0])
            return tf.random.normal(shape, mean=0.0, stddev=stddev, dtype=dtype)
        return initializer

    # Input 9, valid
    input_dict = {
        "categorical_column": categorical_column_3,
        "dimension": 5,
        "combiner": 'sum',
        "initializer": create_initializer_glorot_normal(),
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": None,
        "trainable": True,
        "use_safe_embedding_lookup": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    input_dict = {
        "categorical_column": categorical_column_1,
        "dimension": 7,
        "combiner": 'mean',
        "initializer": create_initializer_zeros(),
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": 0.25,
        "trainable": False,
        "use_safe_embedding_lookup": False
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
