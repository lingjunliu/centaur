
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_embedding_column_inputs():
    list_of_inputs = []

    categorical_column_1 = tf.feature_column.categorical_column_with_identity(num_buckets=10, key='test1')
    categorical_column_2 = tf.feature_column.categorical_column_with_vocabulary_list(vocabulary_list=['a', 'b', 'c'], key='test2')
    categorical_column_3 = tf.feature_column.categorical_column_with_hash_bucket(hash_bucket_size=100, key='test3')
    #categorical_column_4 = tf.feature_column.categorical_column_with_integerized_feature(vocabulary_list=[1, 5, 10], key='test4') # Removed due to AttributeError
    
    def initializer_1(shape, dtype=None, partition_info=None):
        return np.random.normal(size=shape).astype(np.float32)

    def initializer_2(shape, dtype=None, partition_info=None):
        return np.zeros(shape).astype(np.float32)
    
    def initializer_3(shape, dtype=None, partition_info=None):
      return np.ones(shape).astype(np.float32)
    
    ckpt_file = "model.ckpt"
    tensor_name = "embedding_weights"

    # Input 1
    input_dict = {
        "categorical_column": [categorical_column_1],
        "dimension": np.int32(8),
        "combiner": 'mean',
        "initializer": initializer_1,
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": None,
        "trainable": True,
        "use_safe_embedding_lookup": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "categorical_column": [categorical_column_2],
        "dimension": np.int32(16),
        "combiner": 'sqrtn',
        "initializer": initializer_2,
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": np.float32(1.0),
        "trainable": False,
        "use_safe_embedding_lookup": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "categorical_column": [categorical_column_3],
        "dimension": np.int32(4),
        "combiner": 'sum',
        "initializer": initializer_3,
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": np.float32(0.5),
        "trainable": True,
        "use_safe_embedding_lookup": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "categorical_column": [categorical_column_1],
        "dimension": np.int32(32),
        "combiner": 'mean',
        "initializer": initializer_1,
        "ckpt_to_load_from": ckpt_file,
        "tensor_name_in_ckpt": tensor_name,
        "max_norm": None,
        "trainable": False,
        "use_safe_embedding_lookup": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        "categorical_column": [categorical_column_2],
        "dimension": np.int32(64),
        "combiner": 'sqrtn',
        "initializer": initializer_2,
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": np.float32(1.5),
        "trainable": True,
        "use_safe_embedding_lookup": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "categorical_column": [categorical_column_3],
        "dimension": np.int32(2),
        "combiner": 'sum',
        "initializer": initializer_3,
        "ckpt_to_load_from": ckpt_file,
        "tensor_name_in_ckpt": tensor_name,
        "max_norm": np.float32(0.1),
        "trainable": False,
        "use_safe_embedding_lookup": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "categorical_column": [categorical_column_1],
        "dimension": np.int32(128),
        "combiner": 'mean',
        "initializer": initializer_1,
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": np.float32(2.0),
        "trainable": True,
        "use_safe_embedding_lookup": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "categorical_column": [categorical_column_2],
        "dimension": np.int32(1),
        "combiner": 'sqrtn',
        "initializer": initializer_2,
        "ckpt_to_load_from": ckpt_file,
        "tensor_name_in_ckpt": tensor_name,
        "max_norm": None,
        "trainable": False,
        "use_safe_embedding_lookup": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "categorical_column": [categorical_column_3],
        "dimension": np.int32(3),
        "combiner": 'sum',
        "initializer": initializer_3,
        "ckpt_to_load_from": None,
        "tensor_name_in_ckpt": None,
        "max_norm": np.float32(0.01),
        "trainable": True,
        "use_safe_embedding_lookup": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "categorical_column": [categorical_column_1],
        "dimension": np.int32(100),
        "combiner": 'mean',
        "initializer": initializer_1,
        "ckpt_to_load_from": ckpt_file,
        "tensor_name_in_ckpt": tensor_name,
        "max_norm": np.float32(10.0),
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
