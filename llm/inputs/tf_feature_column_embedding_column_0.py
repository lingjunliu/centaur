
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_embedding_column_inputs():
    list_of_inputs = []

    def create_dummy_categorical_column():
        return tf.feature_column.categorical_column_with_vocabulary_list(
            key='test_column',
            vocabulary_list=['a', 'b', 'c'])

    # Input 1
    categorical_column = create_dummy_categorical_column()
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
    categorical_column = create_dummy_categorical_column()
    dimension = 16
    combiner = 'sqrtn'
    initializer = tf.keras.initializers.glorot_uniform()
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
    categorical_column = create_dummy_categorical_column()
    dimension = 4
    combiner = 'sum'
    initializer = tf.keras.initializers.he_normal()
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 0.5
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
    categorical_column = create_dummy_categorical_column()
    dimension = 32
    combiner = 'mean'
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = 2.0
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

    # Input 5
    categorical_column = create_dummy_categorical_column()
    dimension = 1
    combiner = 'mean'
    initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
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
