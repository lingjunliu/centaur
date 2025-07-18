
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_numeric_column_inputs():
    """
    Generates a list of valid inputs for the tf.feature_column.numeric_column function.
    """
    list_of_inputs = []

    # Input 1: Basic case. To satisfy the signature, normalizer_fn is an empty list.
    input_dict_1 = {
        'key': 'price',
        'shape': (1,),
        'default_value': [0.0],
        'dtype': tf.float32,
        'normalizer_fn': []
    }
    list_of_inputs.append(copy.copy(input_dict_1))

    # Input 2: Integer type with a single default value.
    input_dict_2 = {
        'key': 'age',
        'shape': (1,),
        'default_value': [0],
        'dtype': tf.int32,
        'normalizer_fn': []
    }
    list_of_inputs.append(copy.copy(input_dict_2))

    # Input 3: Float type with a normalizer function, wrapped in a list for the signature.
    input_dict_3 = {
        'key': 'temperature_celsius',
        'shape': (1,),
        'default_value': [0.0],
        'dtype': tf.float32,
        'normalizer_fn': [lambda x: (x * 9/5) + 32]
    }
    list_of_inputs.append(copy.copy(input_dict_3))

    # Input 4: Vector feature (shape > 1) with a compatible default value.
    input_dict_4 = {
        'key': 'embedding_vector',
        'shape': (4,),
        'default_value': [0.0, 0.0, 0.0, 0.0],
        'dtype': tf.float64,
        'normalizer_fn': []
    }
    list_of_inputs.append(copy.copy(input_dict_4))

    # Input 5: Vector feature with a default value list.
    input_dict_5 = {
        'key': 'rgb_color',
        'shape': (3,),
        'default_value': [0.0, 0.0, 0.0],
        'dtype': tf.float32,
        'normalizer_fn': []
    }
    list_of_inputs.append(copy.copy(input_dict_5))

    # Input 6: Matrix feature (2D shape) with a compatible default value.
    input_dict_6 = {
        'key': 'image_patch',
        'shape': (2, 2),
        'default_value': [[0, 0], [0, 0]],
        'dtype': tf.int64,
        'normalizer_fn': []
    }
    list_of_inputs.append(copy.copy(input_dict_6))

    # Input 7: Matrix feature with a nested list default value.
    input_dict_7 = {
        'key': 'sensor_grid',
        'shape': (3, 2),
        'default_value': [[0, 0], [0, 0], [0, 0]],
        'dtype': tf.int32,
        'normalizer_fn': []
    }
    list_of_inputs.append(copy.copy(input_dict_7))

    # Input 8: Vector with default and a TensorFlow normalizer function, wrapped in a list.
    input_dict_8 = {
        'key': 'log_scaled_feature',
        'shape': (1,),
        'default_value': [1.0],
        'dtype': tf.float32,
        'normalizer_fn': [lambda x: tf.math.log1p(x)]
    }
    list_of_inputs.append(copy.copy(input_dict_8))

    # Input 9: Using float64 dtype and a negative default value.
    input_dict_9 = {
        'key': 'user_score',
        'shape': (1,),
        'default_value': [-1.0],
        'dtype': tf.float64,
        'normalizer_fn': []
    }
    list_of_inputs.append(copy.copy(input_dict_9))

    # Input 10: Using int64 dtype with a compatible default value.
    input_dict_10 = {
        'key': 'item_id',
        'shape': (1,),
        'default_value': [0],
        'dtype': tf.int64,
        'normalizer_fn': []
    }
    list_of_inputs.append(copy.copy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.feature_column.numeric_column"] = tf_feature_column_numeric_column_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.feature_column.numeric_column' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.numeric_column'.")

check_valid('tf.feature_column.numeric_column', generated_inputs['tf.feature_column.numeric_column'], lib="tf", suffix=0)
