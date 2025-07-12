
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_numeric_column_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'key': 'feature_a',
        'shape': (1,),
        'default_value': [0.0],
        'dtype': tf.float32,
        'normalizer_fn': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'key': 'feature_b',
        'shape': (2, 2),
        'default_value': [1.0, 2.0, 3.0, 4.0],
        'dtype': tf.float64,
        'normalizer_fn': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'key': 'feature_c',
        'shape': (3,),
        'default_value': [-1, 0, 1],
        'dtype': tf.int32,
        'normalizer_fn': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'key': 'feature_d',
        'shape': (1, 1, 1),
        'default_value': [2.5],
        'dtype': tf.float16,
        'normalizer_fn': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'key': 'feature_e',
        'shape': (4,),
        'default_value': [1, 2, 3, 4],
        'dtype': tf.int64,
        'normalizer_fn': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    input_dict = {
        'key': 'feature_f',
        'shape': (2,),
        'default_value': [-1.5, 2.5],
        'dtype': tf.float32,
        'normalizer_fn': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'key': 'feature_g',
        'shape': (1,),
        'default_value': [1000],
        'dtype': tf.int16,
        'normalizer_fn': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'key': 'feature_h',
        'shape': (5,),
        'default_value': [0.1, 0.2, 0.3, 0.4, 0.5],
        'dtype': tf.float32,
        'normalizer_fn': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'key': 'feature_i',
        'shape': (1, 2),
        'default_value': [5, 6],
        'dtype': tf.int8,
        'normalizer_fn': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'key': 'feature_j',
        'shape': (2, 1),
        'default_value': [7, 8],
        'dtype': tf.int32,
        'normalizer_fn': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 with normalizer_fn
    input_dict = {
        'key': 'feature_k',
        'shape': (1,),
        'default_value': [2.0],
        'dtype': tf.float32,
        'normalizer_fn': lambda x: x / 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 with normalizer_fn that returns different type
    input_dict = {
        'key': 'feature_l',
        'shape': (1,),
        'default_value': [5.0],
        'dtype': tf.float32,
        'normalizer_fn': lambda x: tf.cast(x, tf.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13 with normalizer fn
    input_dict = {
        'key': 'feature_m',
        'shape': (1,),
        'default_value': [10.0],
        'dtype': tf.float32,
        'normalizer_fn': lambda x: (x - 5.0) / 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.numeric_column"] = tf_feature_column_numeric_column_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.numeric_column' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.numeric_column'.")

check_valid('tf.feature_column.numeric_column', generated_inputs['tf.feature_column.numeric_column'], lib="tf", suffix=0)
