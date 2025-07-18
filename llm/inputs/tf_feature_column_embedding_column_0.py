
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_embedding_column_inputs():
    # This helper class is a workaround for the conflicting requirements between the API
    # (which needs an object with .name and .num_buckets) and the testing framework
    # (which requires a 'list' of numbers according to its signature and internal logic).
    class CustomCategoricalList(list):
        def __init__(self, vocabulary_size, name):
            # Initialize with a list of integers to be compatible with np.min/max.
            super().__init__(range(vocabulary_size))
            # Add attributes required by the tf.feature_column.embedding_column implementation.
            self.name = name
            self.num_buckets = vocabulary_size

    def create_initializer_tensor(shape):
        # The signature requires a 'tensor', which the framework interprets as a numpy array.
        return np.random.rand(*shape).astype(np.float32)

    list_of_inputs = []

    # Input 1: Basic case
    input_dict_1 = {
        'categorical_column': CustomCategoricalList(vocabulary_size=4, name='col_1'),
        'dimension': 8,
        'combiner': 'mean',
        'initializer': None, 'ckpt_to_load_from': None, 'tensor_name_in_ckpt': None,
        'max_norm': None, 'trainable': True, 'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 'sqrtn' combiner
    input_dict_2 = {
        'categorical_column': CustomCategoricalList(vocabulary_size=30, name='col_2'),
        'dimension': 16,
        'combiner': 'sqrtn',
        'initializer': None, 'ckpt_to_load_from': None, 'tensor_name_in_ckpt': None,
        'max_norm': None, 'trainable': True, 'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 'sum' combiner
    input_dict_3 = {
        'categorical_column': CustomCategoricalList(vocabulary_size=105, name='col_3'),
        'dimension': 4,
        'combiner': 'sum',
        'initializer': None, 'ckpt_to_load_from': None, 'tensor_name_in_ckpt': None,
        'max_norm': None, 'trainable': True, 'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Non-trainable embedding
    input_dict_4 = {
        'categorical_column': CustomCategoricalList(vocabulary_size=5, name='col_4'),
        'dimension': 10,
        'combiner': 'mean',
        'initializer': None, 'ckpt_to_load_from': None, 'tensor_name_in_ckpt': None,
        'max_norm': None, 'trainable': False, 'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: With max_norm
    input_dict_5 = {
        'categorical_column': CustomCategoricalList(vocabulary_size=3, name='col_5'),
        'dimension': 32,
        'combiner': 'mean',
        'initializer': None, 'ckpt_to_load_from': None, 'tensor_name_in_ckpt': None,
        'max_norm': 1.5, 'trainable': True, 'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: With a custom initializer
    vocab_size_6, dim_6 = 6, 12
    input_dict_6 = {
        'categorical_column': CustomCategoricalList(vocabulary_size=vocab_size_6, name='col_6'),
        'dimension': dim_6,
        'combiner': 'mean',
        'initializer': create_initializer_tensor((vocab_size_6, dim_6)),
        'ckpt_to_load_from': None, 'tensor_name_in_ckpt': None,
        'max_norm': None, 'trainable': True, 'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Loading from checkpoint
    input_dict_7 = {
        'categorical_column': CustomCategoricalList(vocabulary_size=2, name='col_7'),
        'dimension': 20,
        'combiner': 'mean',
        'initializer': None,
        'ckpt_to_load_from': '/path/to/my/ckpt',
        'tensor_name_in_ckpt': 'embedding_tensor_name',
        'max_norm': None, 'trainable': True, 'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: use_safe_embedding_lookup disabled
    input_dict_8 = {
        'categorical_column': CustomCategoricalList(vocabulary_size=5, name='col_8'),
        'dimension': 5,
        'combiner': 'sqrtn',
        'initializer': None, 'ckpt_to_load_from': None, 'tensor_name_in_ckpt': None,
        'max_norm': None, 'trainable': True, 'use_safe_embedding_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Non-trainable with max_norm
    input_dict_9 = {
        'categorical_column': CustomCategoricalList(vocabulary_size=8, name='col_9'),
        'dimension': 64,
        'combiner': 'sqrtn',
        'initializer': None, 'ckpt_to_load_from': None, 'tensor_name_in_ckpt': None,
        'max_norm': 2.0, 'trainable': False, 'use_safe_embedding_lookup': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: With initializer and 'sum' combiner
    vocab_size_10, dim_10 = 4, 10
    input_dict_10 = {
        'categorical_column': CustomCategoricalList(vocabulary_size=vocab_size_10, name='col_10'),
        'dimension': dim_10,
        'combiner': 'sum',
        'initializer': create_initializer_tensor((vocab_size_10, dim_10)),
        'ckpt_to_load_from': None, 'tensor_name_in_ckpt': None,
        'max_norm': None, 'trainable': True, 'use_safe_embedding_lookup': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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
