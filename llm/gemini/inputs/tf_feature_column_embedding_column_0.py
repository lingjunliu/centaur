
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_embedding_column_inputs():
    list_of_inputs = []

    categorical_column = [tf.feature_column.categorical_column_with_identity(key='test', num_buckets=5)]
    initializer = np.random.normal(0.0, 1.0, size=(8,)).astype(np.float32)

    input_dict = {
        "categorical_column": categorical_column,
        "dimension": 8,
        "combiner": "mean",
        "initializer": initializer,
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": None,
        "trainable": True,
        "use_safe_embedding_lookup": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_list(key='test2', vocabulary_list=['a', 'b', 'c'])]
    initializer = np.random.uniform(size=(16,)).astype(np.float32)

    input_dict = {
        "categorical_column": categorical_column,
        "dimension": 16,
        "combiner": "sqrtn",
        "initializer": initializer,
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": 1.0,
        "trainable": False,
        "use_safe_embedding_lookup": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(key='test3', hash_bucket_size=100)]
    initializer = np.zeros((4,)).astype(np.float32)

    input_dict = {
        "categorical_column": categorical_column,
        "dimension": 4,
        "combiner": "sum",
        "initializer": initializer,
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": None,
        "trainable": True,
        "use_safe_embedding_lookup": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    categorical_column = [tf.feature_column.categorical_column_with_identity(key='test4', num_buckets=2)]
    initializer = np.ones((32,)).astype(np.float32)

    input_dict = {
        "categorical_column": categorical_column,
        "dimension": 32,
        "combiner": "mean",
        "initializer": initializer,
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": None,
        "trainable": True,
        "use_safe_embedding_lookup": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(key='test7', hash_bucket_size=50)]
    initializer = np.random.rand(2,).astype(np.float32)

    input_dict = {
        "categorical_column": categorical_column,
        "dimension": 2,
        "combiner": "mean",
        "initializer": initializer,
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": 0.1,
        "trainable": False,
        "use_safe_embedding_lookup": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_list(key='test8', vocabulary_list=['x', 'y', 'z', 'w'])]
    initializer = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)
    input_dict = {
        "categorical_column": categorical_column,
        "dimension": 5,
        "combiner": "sqrtn",
        "initializer": initializer,
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": 5.0,
        "trainable": True,
        "use_safe_embedding_lookup": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    categorical_column = [tf.feature_column.categorical_column_with_identity(key='test9', num_buckets=3)]
    initializer = np.random.randn(10,).astype(np.float32)
    input_dict = {
        "categorical_column": categorical_column,
        "dimension": 10,
        "combiner": "sum",
        "initializer": initializer,
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": 10.0,
        "trainable": False,
        "use_safe_embedding_lookup": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.embedding_column"] = tf_feature_column_embedding_column_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.feature_column.embedding_column' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.embedding_column'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.feature_column.embedding_column', generated_inputs['tf.feature_column.embedding_column'], lib="tf", suffix=0)
