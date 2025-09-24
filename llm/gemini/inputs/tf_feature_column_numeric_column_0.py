
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_numeric_column_inputs():
    """
    Generates a list of valid inputs for tf.feature_column.numeric_column.
    """
    list_of_inputs = []

    # Input 1: Basic case, no optional args used
    input_1 = {
        'key': 'price',
        'shape': (1,),
        'default_value': None,
        'dtype': np.dtype('float32'),
        'normalizer_fn': None
    }
    list_of_inputs.append(copy.deepcopy(input_1))

    # Input 2: Basic case with int32
    input_2 = {
        'key': 'age',
        'shape': (1,),
        'default_value': None,
        'dtype': np.dtype('int32'),
        'normalizer_fn': None
    }
    list_of_inputs.append(copy.deepcopy(input_2))

    # Input 3: Vector feature with float64
    input_3 = {
        'key': 'embedding_vector',
        'shape': (4,),
        'default_value': None,
        'dtype': np.dtype('float64'),
        'normalizer_fn': None
    }
    list_of_inputs.append(copy.deepcopy(input_3))

    # Input 4: Scalar feature with a default value
    input_4 = {
        'key': 'temperature',
        'shape': (1,),
        'default_value': [-1.0],
        'dtype': np.dtype('float32'),
        'normalizer_fn': None
    }
    list_of_inputs.append(copy.deepcopy(input_4))

    # Input 5: Vector feature with a default value
    input_5 = {
        'key': 'rgb_values',
        'shape': (3,),
        'default_value': [0, 0, 0],
        'dtype': np.dtype('int32'),
        'normalizer_fn': None
    }
    list_of_inputs.append(copy.deepcopy(input_5))

    # Input 6: With a normalizer function
    input_6 = {
        'key': 'score',
        'shape': (1,),
        'default_value': None,
        'dtype': np.dtype('float32'),
        'normalizer_fn': lambda x: (x - 50.0) / 100.0
    }
    list_of_inputs.append(copy.deepcopy(input_6))

    # Input 7: Multi-dimensional shape (2D)
    input_7 = {
        'key': 'image_patch',
        'shape': (2, 2),
        'default_value': None,
        'dtype': np.dtype('float32'),
        'normalizer_fn': None
    }
    list_of_inputs.append(copy.deepcopy(input_7))

    # Input 8: Multi-dimensional shape with a default value
    input_8 = {
        'key': 'matrix_feature',
        'shape': (2, 3),
        'default_value': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        'dtype': np.dtype('float64'),
        'normalizer_fn': None
    }
    list_of_inputs.append(copy.deepcopy(input_8))

    # Input 9: int64 dtype with a default value
    input_9 = {
        'key': 'user_id',
        'shape': (1,),
        'default_value': [-1],
        'dtype': np.dtype('int64'),
        'normalizer_fn': None
    }
    list_of_inputs.append(copy.deepcopy(input_9))

    # Input 10: Full combination of parameters
    input_10 = {
        'key': 'pixel_data',
        'shape': (3, 3),
        'default_value': [[0] * 3] * 3,
        'dtype': np.dtype('int32'),
        'normalizer_fn': lambda t: tf.cast(t, tf.float32) / 255.0
    }
    list_of_inputs.append(copy.deepcopy(input_10))

    # Input 11: Another complex normalizer with log transform
    input_11 = {
        'key': 'view_counts',
        'shape': (1,),
        'default_value': [0],
        'dtype': np.dtype('int64'),
        'normalizer_fn': lambda x: tf.math.log1p(tf.cast(x, tf.float32))
    }
    list_of_inputs.append(copy.deepcopy(input_11))

    # Input 12: Higher dimensional shape with normalizer
    input_12 = {
        'key': 'sensor_grid',
        'shape': (8, 8),
        'default_value': None,
        'dtype': np.dtype('float32'),
        'normalizer_fn': lambda x: x - tf.reduce_mean(x)
    }
    list_of_inputs.append(copy.deepcopy(input_12))


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
