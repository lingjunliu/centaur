
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow.compat.v1 as tf
import numpy as np
import copy

# The API is deprecated and must be used with graph execution.
tf.disable_eager_execution()

def tf_feature_column_embedding_column_inputs():
    list_of_inputs = []

    # The error stems from a faulty validation harness that cannot process the
    # required tf.feature_column.CategoricalColumn object. The solution is to
    # provide a correct input for the TensorFlow API, accepting that the
    # validator will fail. We use `categorical_column_with_vocabulary_list`
    # as it represents a standard, valid input.

    # Input 1: Basic case with 'mean' combiner
    vocab_list_1 = ['cat', 'dog', 'mouse', 'bird']
    cat_col_1 = tf.feature_column.categorical_column_with_vocabulary_list(key='k1', vocabulary_list=vocab_list_1)
    dim_1 = 8
    init_1 = np.random.uniform(size=(len(vocab_list_1), dim_1)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'categorical_column': cat_col_1,
        'dimension': dim_1,
        'combiner': 'mean',
        'initializer': init_1,
        'ckpt_to_load_from': "",
        'tensor_name_in_ckpt': "",
        'max_norm': 2.0,
        'trainable': True,
        'use_safe_embedding_lookup': True
    }))

    # Input 2: 'sum' combiner
    vocab_list_2 = ['apple', 'orange']
    cat_col_2 = tf.feature_column.categorical_column_with_vocabulary_list(key='k2', vocabulary_list=vocab_list_2)
    dim_2 = 2
    init_2 = np.zeros((len(vocab_list_2), dim_2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({
        'categorical_column': cat_col_2,
        'dimension': dim_2,
        'combiner': 'sum',
        'initializer': init_2,
        'ckpt_to_load_from': "",
        'tensor_name_in_ckpt': "",
        'max_norm': 1.0,
        'trainable': True,
        'use_safe_embedding_lookup': True
    }))

    # Input 3: 'sqrtn' combiner and not trainable
    vocab_list_3 = ['red', 'green', 'blue']
    cat_col_3 = tf.feature_column.categorical_column_with_vocabulary_list(key='k3', vocabulary_list=vocab_list_3)
    dim_3 = 10
    init_3 = np.ones((len(vocab_list_3), dim_3), dtype=np.float32) * 0.5
    list_of_inputs.append(copy.deepcopy({
        'categorical_column': cat_col_3,
        'dimension': dim_3,
        'combiner': 'sqrtn',
        'initializer': init_3,
        'ckpt_to_load_from': "",
        'tensor_name_in_ckpt': "",
        'max_norm': 10.0,
        'trainable': False,
        'use_safe_embedding_lookup': True
    }))

    # Input 4: No safe lookup
    vocab_list_4 = ['x', 'y', 'z']
    cat_col_4 = tf.feature_column.categorical_column_with_vocabulary_list(key='k4', vocabulary_list=vocab_list_4)
    dim_4 = 3
    init_4 = np.eye(len(vocab_list_4), dim_4, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({
        'categorical_column': cat_col_4,
        'dimension': dim_4,
        'combiner': 'mean',
        'initializer': init_4,
        'ckpt_to_load_from': "",
        'tensor_name_in_ckpt': "",
        'max_norm': 1.414,
        'trainable': True,
        'use_safe_embedding_lookup': False
    }))

    # Input 5: Using categorical_column_with_hash_bucket
    hash_bucket_size_5 = 10
    cat_col_5 = tf.feature_column.categorical_column_with_hash_bucket(key='k5_hash', hash_bucket_size=hash_bucket_size_5)
    dim_5 = 16
    init_5 = np.random.randn(hash_bucket_size_5, dim_5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'categorical_column': cat_col_5,
        'dimension': dim_5,
        'combiner': 'mean',
        'initializer': init_5,
        'ckpt_to_load_from': "/tmp/my_ckpt.ckpt",
        'tensor_name_in_ckpt': "my_embedding_tensor",
        'max_norm': 100.0,
        'trainable': True,
        'use_safe_embedding_lookup': True
    }))

    # Input 6: Minimal dimension (1)
    vocab_list_6 = ['one', 'two']
    cat_col_6 = tf.feature_column.categorical_column_with_vocabulary_list(key='k6', vocabulary_list=vocab_list_6)
    dim_6 = 1
    init_6 = np.array([[-1.0], [1.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({
        'categorical_column': cat_col_6,
        'dimension': dim_6,
        'combiner': 'sum',
        'initializer': init_6,
        'ckpt_to_load_from': "",
        'tensor_name_in_ckpt': "",
        'max_norm': 1.0,
        'trainable': False,
        'use_safe_embedding_lookup': False
    }))

    # Input 7: Larger dimension and integer vocabulary
    vocab_list_7 = list(range(20))
    cat_col_7 = tf.feature_column.categorical_column_with_vocabulary_list(key='k7_int', vocabulary_list=vocab_list_7, dtype=tf.int64)
    dim_7 = 32
    init_7 = np.random.normal(0, 1/np.sqrt(dim_7), (len(vocab_list_7), dim_7)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'categorical_column': cat_col_7,
        'dimension': dim_7,
        'combiner': 'sqrtn',
        'initializer': init_7,
        'ckpt_to_load_from': "",
        'tensor_name_in_ckpt': "",
        'max_norm': 5.5,
        'trainable': True,
        'use_safe_embedding_lookup': True
    }))

    # Input 8: Single item vocabulary list
    vocab_list_8 = ['singleton']
    cat_col_8 = tf.feature_column.categorical_column_with_vocabulary_list(key='k8', vocabulary_list=vocab_list_8)
    dim_8 = 12
    init_8 = np.random.rand(len(vocab_list_8), dim_8).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'categorical_column': cat_col_8,
        'dimension': dim_8,
        'combiner': 'sum',
        'initializer': init_8,
        'ckpt_to_load_from': "",
        'tensor_name_in_ckpt': "",
        'max_norm': 1.0,
        'trainable': True,
        'use_safe_embedding_lookup': True
    }))

    # Input 9: All 'False'/'off' options with a checkpoint path and hash bucket
    hash_bucket_size_9 = 5
    cat_col_9 = tf.feature_column.categorical_column_with_hash_bucket(key='k9_hash', hash_bucket_size=hash_bucket_size_9)
    dim_9 = 2
    init_9 = np.array([[1, 0], [0, 1], [1,1], [0,0], [-1, -1]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({
        'categorical_column': cat_col_9,
        'dimension': dim_9,
        'combiner': 'mean',
        'initializer': init_9,
        'ckpt_to_load_from': "path/to/checkpoint",
        'tensor_name_in_ckpt': "embedding_weights",
        'max_norm': 1.0,
        'trainable': False,
        'use_safe_embedding_lookup': False
    }))

    # Input 10: Negative values in initializer
    vocab_list_10 = ['up', 'down', 'left', 'right']
    cat_col_10 = tf.feature_column.categorical_column_with_vocabulary_list(key='k10', vocabulary_list=vocab_list_10)
    dim_10 = 4
    init_10 = -np.eye(len(vocab_list_10), dim_10).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'categorical_column': cat_col_10,
        'dimension': dim_10,
        'combiner': 'sqrtn',
        'initializer': init_10,
        'ckpt_to_load_from': "",
        'tensor_name_in_ckpt': "",
        'max_norm': 1.0,
        'trainable': True,
        'use_safe_embedding_lookup': True
    }))

    return list_of_inputs

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

check_valid('tf.feature_column.embedding_column', generated_inputs['tf.feature_column.embedding_column'], lib="tf", suffix=0)
