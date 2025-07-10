
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_embedding_column_inputs():
    list_of_inputs = []

    def create_initializer(dimension, num_buckets):
        return tf.constant(tf.compat.v1.truncated_normal(shape=(num_buckets, dimension), mean=0.0, stddev=1.0))

    # Input 1
    categorical_column = [tf.feature_column.categorical_column_with_identity(key='test', num_buckets=10)]
    dimension = 8
    combiner = 'mean'
    initializer = create_initializer(dimension, 10)
    ckpt_to_load_from = ''
    tensor_name_in_ckpt = ''
    max_norm = None
    trainable = True
    use_safe_embedding_lookup = True

    input_dict = {
        'categorical_column': categorical_column,
        'dimension': dimension,
        'combiner': combiner,
        'initializer': initializer,
        'ckpt_to_load_from': ckpt_to_load_from,
        'tensor_name_in_ckpt': tensor_name_in_ckpt,
        'max_norm': max_norm,
        'trainable': trainable,
        'use_safe_embedding_lookup': use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_list(key='colors', vocabulary_list=['red', 'green', 'blue'])]
    dimension = 16
    combiner = 'sqrtn'
    initializer = create_initializer(dimension, 3)
    ckpt_to_load_from = ''
    tensor_name_in_ckpt = ''
    max_norm = 1.0
    trainable = False
    use_safe_embedding_lookup = False

    input_dict = {
        'categorical_column': categorical_column,
        'dimension': dimension,
        'combiner': combiner,
        'initializer': initializer,
        'ckpt_to_load_from': ckpt_to_load_from,
        'tensor_name_in_ckpt': tensor_name_in_ckpt,
        'max_norm': max_norm,
        'trainable': trainable,
        'use_safe_embedding_lookup': use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(key='text', hash_bucket_size=1000)]
    dimension = 4
    combiner = 'sum'
    initializer = create_initializer(dimension, 1000)
    ckpt_to_load_from = ''
    tensor_name_in_ckpt = ''
    max_norm = 0.5
    trainable = True
    use_safe_embedding_lookup = True

    input_dict = {
        'categorical_column': categorical_column,
        'dimension': dimension,
        'combiner': combiner,
        'initializer': initializer,
        'ckpt_to_load_from': ckpt_to_load_from,
        'tensor_name_in_ckpt': tensor_name_in_ckpt,
        'max_norm': max_norm,
        'trainable': trainable,
        'use_safe_embedding_lookup': use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 4
    categorical_column = [tf.feature_column.categorical_column_with_identity(key='id', num_buckets=50)]
    dimension = 32
    combiner = 'mean'
    initializer = create_initializer(dimension, 50)
    ckpt_to_load_from = ''
    tensor_name_in_ckpt = ''
    max_norm = 2.0
    trainable = False
    use_safe_embedding_lookup = False

    input_dict = {
        'categorical_column': categorical_column,
        'dimension': dimension,
        'combiner': combiner,
        'initializer': initializer,
        'ckpt_to_load_from': ckpt_to_load_from,
        'tensor_name_in_ckpt': tensor_name_in_ckpt,
        'max_norm': max_norm,
        'trainable': trainable,
        'use_safe_embedding_lookup': use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_file(key='words', vocabulary_file='vocab.txt', vocabulary_size=100)]

    dimension = 64
    combiner = 'sqrtn'
    initializer = create_initializer(dimension, 100)
    ckpt_to_load_from = ''
    tensor_name_in_ckpt = ''
    max_norm = None
    trainable = True
    use_safe_embedding_lookup = True

    input_dict = {
        'categorical_column': categorical_column,
        'dimension': dimension,
        'combiner': combiner,
        'initializer': initializer,
        'ckpt_to_load_from': ckpt_to_load_from,
        'tensor_name_in_ckpt': tensor_name_in_ckpt,
        'max_norm': max_norm,
        'trainable': trainable,
        'use_safe_embedding_lookup': use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(key='feature', hash_bucket_size=2000)]
    dimension = 128
    combiner = 'sum'
    initializer = create_initializer(dimension, 2000)
    ckpt_to_load_from = ''
    tensor_name_in_ckpt = ''
    max_norm = 1.5
    trainable = False
    use_safe_embedding_lookup = False

    input_dict = {
        'categorical_column': categorical_column,
        'dimension': dimension,
        'combiner': combiner,
        'initializer': initializer,
        'ckpt_to_load_from': ckpt_to_load_from,
        'tensor_name_in_ckpt': tensor_name_in_ckpt,
        'max_norm': max_norm,
        'trainable': trainable,
        'use_safe_embedding_lookup': use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    categorical_column = [tf.feature_column.categorical_column_with_identity(key='small_id', num_buckets=2)]
    dimension = 2
    combiner = 'mean'
    initializer = create_initializer(dimension, 2)
    ckpt_to_load_from = ''
    tensor_name_in_ckpt = ''
    max_norm = 0.1
    trainable = True
    use_safe_embedding_lookup = True

    input_dict = {
        'categorical_column': categorical_column,
        'dimension': dimension,
        'combiner': combiner,
        'initializer': initializer,
        'ckpt_to_load_from': ckpt_to_load_from,
        'tensor_name_in_ckpt': tensor_name_in_ckpt,
        'max_norm': max_norm,
        'trainable': trainable,
        'use_safe_embedding_lookup': use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    categorical_column = [tf.feature_column.categorical_column_with_vocabulary_list(key='symbols', vocabulary_list=['a', 'b', 'c', 'd', 'e'])]
    dimension = 10
    combiner = 'sqrtn'
    initializer = create_initializer(dimension, 5)
    ckpt_to_load_from = ''
    tensor_name_in_ckpt = ''
    max_norm = None
    trainable = False
    use_safe_embedding_lookup = False

    input_dict = {
        'categorical_column': categorical_column,
        'dimension': dimension,
        'combiner': combiner,
        'initializer': initializer,
        'ckpt_to_load_from': ckpt_to_load_from,
        'tensor_name_in_ckpt': tensor_name_in_ckpt,
        'max_norm': max_norm,
        'trainable': trainable,
        'use_safe_embedding_lookup': use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    categorical_column = [tf.feature_column.categorical_column_with_identity(key='bucket_id', num_buckets=100)]
    dimension = 256
    combiner = 'sum'
    initializer = create_initializer(dimension, 100)
    ckpt_to_load_from = ''
    tensor_name_in_ckpt = ''
    max_norm = 3.0
    trainable = True
    use_safe_embedding_lookup = True

    input_dict = {
        'categorical_column': categorical_column,
        'dimension': dimension,
        'combiner': combiner,
        'initializer': initializer,
        'ckpt_to_load_from': ckpt_to_load_from,
        'tensor_name_in_ckpt': tensor_name_in_ckpt,
        'max_norm': max_norm,
        'trainable': trainable,
        'use_safe_embedding_lookup': use_safe_embedding_lookup
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    categorical_column = [tf.feature_column.categorical_column_with_hash_bucket(key='long_feature', hash_bucket_size=10000)]
    dimension = 512
    combiner = 'mean'
    initializer = create_initializer(dimension, 10000)
    ckpt_to_load_from = ''
    tensor_name_in_ckpt = ''
    max_norm = None
    trainable = False
    use_safe_embedding_lookup = False

    input_dict = {
        'categorical_column': categorical_column,
        'dimension': dimension,
        'combiner': combiner,
        'initializer': initializer,
        'ckpt_to_load_from': ckpt_to_load_from,
        'tensor_name_in_ckpt': tensor_name_in_ckpt,
        'max_norm': max_norm,
        'trainable': trainable,
        'use_safe_embedding_lookup': use_safe_embedding_lookup
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
