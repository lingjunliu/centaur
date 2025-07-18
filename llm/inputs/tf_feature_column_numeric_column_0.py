
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_feature_column_numeric_column_inputs():
    """
    Generates a list of valid inputs for the tf.feature_column.numeric_column function.
    """
    list_of_inputs = []

    # Input 1: Basic case with float32
    input_dict_1 = {
        'key': 'feature_a',
        'shape': (1,),
        'default_value': [0.0],
        'dtype': tf.float32,
        'normalizer_fn': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic case with int32
    input_dict_2 = {
        'key': 'feature_b',
        'shape': (1,),
        'default_value': [0],
        'dtype': tf.int32,
        'normalizer_fn': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: With a different single default value
    input_dict_3 = {
        'key': 'price',
        'shape': (1,),
        'default_value': [-99.0],
        'dtype': tf.float32,
        'normalizer_fn': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: With a simple normalizer function
    input_dict_4 = {
        'key': 'temperature',
        'shape': (1,),
        'default_value': [32.0],
        'dtype': tf.float32,
        'normalizer_fn': [lambda x: (x - 32.0) * 5.0/9.0]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Multi-dimensional shape (1D vector) with a default value
    input_dict_5 = {
        'key': 'vector_feature',
        'shape': (4,),
        'default_value': [0.0, 0.0, 0.0, 0.0],
        'dtype': tf.float32,
        'normalizer_fn': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Multi-dimensional shape (1D vector) with a different default value
    input_dict_6 = {
        'key': 'embedding',
        'shape': (3,),
        'default_value': [0.1, 0.2, 0.3],
        'dtype': tf.float32,
        'normalizer_fn': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 2D shape with a 2D default value and float64 dtype
    input_dict_7 = {
        'key': 'image_patch',
        'shape': (2, 2),
        'default_value': [[1.0, 0.0], [0.0, 1.0]],
        'dtype': tf.float64,
        'normalizer_fn': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: All arguments used, with negative default value
    input_dict_8 = {
        'key': 'sensor_readings',
        'shape': (3,),
        'default_value': [-1.0, -1.0, -1.0],
        'dtype': tf.float32,
        'normalizer_fn': [lambda x: (x - 100.0) / 50.0]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Integer type with default value and a normalizer that casts to float
    input_dict_9 = {
        'key': 'age',
        'shape': (1,),
        'default_value': [0],
        'dtype': tf.int64,
        'normalizer_fn': [lambda x: tf.cast(x, tf.float32) / 100.0]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 2D shape with a single default value to be broadcasted
    input_dict_10 = {
        'key': 'matrix_feature',
        'shape': (3, 2),
        'default_value': [0.0],
        'dtype': tf.float32,
        'normalizer_fn': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: With a more complex normalizer (log1p)
    input_dict_11 = {
        'key': 'log_feature',
        'shape': (1,),
        'default_value': [0.0],
        'dtype': tf.float32,
        'normalizer_fn': [lambda x: tf.math.log1p(x)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.numeric_column"] = get_tf_feature_column_numeric_column_inputs()

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
