
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# This API is part of the deprecated tf.feature_column module and is designed
# for TensorFlow's graph mode. To create the required CategoricalColumn objects
# without a RuntimeError, eager execution must be disabled.
tf.compat.v1.disable_eager_execution()

def get_tf_feature_column_embedding_column_inputs():
    list_of_inputs = []

    # NOTE: To fix the `AttributeError: 'list' object has no attribute 'name'`,
    # the `categorical_column` parameter must be a `CategoricalColumn` object,
    # as required by the TensorFlow API, not a Python list. The following inputs
    # are generated according to this API requirement.

    def create_initializer_array(shape):
        return np.random.rand(*shape).astype(np.float32)

    # Input 1: Basic case with vocabulary list
    cat_col_1 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='feature1', vocabulary_list=['a', 'b', 'c'])
    input_dict_1 = {
        'categorical_column': cat_col_1,
        'dimension': 8,
        'combiner': 'mean',
        'initializer': None,
        'ckpt_to_load_from': None,
        'tensor_name_in_ckpt': None,
        'max_norm': None,
        'trainable': True,
        'use_safe_embedding_lookup': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Using 'sum' combiner
    cat_col_2 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='feature2', vocabulary_list=['cat', 'dog', 'mouse', 'bird'])
    input_dict_2 = {
        'categorical_column': cat_col_2,
        'dimension': 16,
        'combiner': 'sum',
        'initializer': None,
        'ckpt_to_load_from': None,
        'tensor_name_in_ckpt': None,
        'max_norm': None,
        'trainable': True,
        'use_safe_embedding_lookup': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Using 'sqrtn' combiner
    cat_col_3 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='feature3', vocabulary_list=['x', 'y'])
    input_dict_3 = {
        'categorical_column': cat_col_3,
        'dimension': 4,
        'combiner': 'sqrtn',
        'initializer': None,
        'ckpt_to_load_from': None,
        'tensor_name_in_ckpt': None,
        'max_norm': None,
        'trainable': True,
        'use_safe_embedding_lookup': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Using hash bucket
    cat_col_4 = tf.feature_column.categorical_column_with_hash_bucket(
        key='feature4', hash_bucket_size=10)
    input_dict_4 = {
        'categorical_column': cat_col_4,
        'dimension': 32,
        'combiner': 'mean',
        'initializer': None,
        'ckpt_to_load_from': None,
        'tensor_name_in_ckpt': None,
        'max_norm': None,
        'trainable': True,
        'use_safe_embedding_lookup': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: With max_norm
    cat_col_5 = tf.feature_column.categorical_column_with_hash_bucket(
        key='feature5', hash_bucket_size=100)
    input_dict_5 = {
        'categorical_column': cat_col_5,
        'dimension': 64,
        'combiner': 'sqrtn',
        'initializer': None,
        'ckpt_to_load_from': None,
        'tensor_name_in_ckpt': None,
        'max_norm': 1.5,
        'trainable': True,
        'use_safe_embedding_lookup': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Not trainable
    cat_col_6 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='feature6', vocabulary_list=['on', 'off'])
    input_dict_6 = {
        'categorical_column': cat_col_6,
        'dimension': 2,
        'combiner': 'mean',
        'initializer': None,
        'ckpt_to_load_from': None,
        'tensor_name_in_ckpt': None,
        'max_norm': None,
        'trainable': False,
        'use_safe_embedding_lookup': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: use_safe_embedding_lookup set to False
    cat_col_7 = tf.feature_column.categorical_column_with_hash_bucket(
        key='feature7', hash_bucket_size=50)
    input_dict_7 = {
        'categorical_column': cat_col_7,
        'dimension': 10,
        'combiner': 'sum',
        'initializer': None,
        'ckpt_to_load_from': None,
        'tensor_name_in_ckpt': None,
        'max_norm': None,
        'trainable': True,
        'use_safe_embedding_lookup': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: With a custom initializer for vocabulary list
    vocab_list_8 = ['apple', 'banana', 'cherry', 'date']
    dimension_8 = 5
    cat_col_8 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='feature8', vocabulary_list=vocab_list_8)
    input_dict_8 = {
        'categorical_column': cat_col_8,
        'dimension': dimension_8,
        'combiner': 'mean',
        'initializer': create_initializer_array((len(vocab_list_8), dimension_8)),
        'ckpt_to_load_from': None,
        'tensor_name_in_ckpt': None,
        'max_norm': None,
        'trainable': True,
        'use_safe_embedding_lookup': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Loading from checkpoint
    cat_col_9 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='feature9', vocabulary_list=['red', 'green', 'blue'])
    input_dict_9 = {
        'categorical_column': cat_col_9,
        'dimension': 3,
        'combiner': 'mean',
        'initializer': None,
        'ckpt_to_load_from': '/tmp/my_model.ckpt',
        'tensor_name_in_ckpt': 'feature9_embedding/embedding_weights',
        'max_norm': None,
        'trainable': True,
        'use_safe_embedding_lookup': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Complex combination with hash bucket and initializer
    hash_bucket_size_10 = 1000
    dimension_10 = 128
    cat_col_10 = tf.feature_column.categorical_column_with_hash_bucket(
        key='feature10', hash_bucket_size=hash_bucket_size_10)
    input_dict_10 = {
        'categorical_column': cat_col_10,
        'dimension': dimension_10,
        'combiner': 'sqrtn',
        'initializer': create_initializer_array((hash_bucket_size_10, dimension_10)),
        'ckpt_to_load_from': None,
        'tensor_name_in_ckpt': None,
        'max_norm': 2.0,
        'trainable': False,
        'use_safe_embedding_lookup': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Minimal valid dimension
    cat_col_11 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='feature11', vocabulary_list=['true', 'false', 'unknown'])
    input_dict_11 = {
        'categorical_column': cat_col_11,
        'dimension': 1,
        'combiner': 'mean',
        'initializer': None,
        'ckpt_to_load_from': None,
        'tensor_name_in_ckpt': None,
        'max_norm': 0.5,
        'trainable': True,
        'use_safe_embedding_lookup': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.feature_column.embedding_column"] = get_tf_feature_column_embedding_column_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.feature_column.embedding_column' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.embedding_column'.")

check_valid('tf.feature_column.embedding_column', generated_inputs['tf.feature_column.embedding_column'], lib="tf", suffix=0)
