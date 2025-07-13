
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_numeric_column_inputs():
    list_of_inputs = []

    def no_op_normalizer(x):
        return x

    # Input 1: Basic valid input
    input_dict = {
        'key': 'feature_a',
        'shape': (1,),
        'default_value': [0.0],
        'dtype': tf.float32,
        'normalizer_fn': [no_op_normalizer]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shape
    input_dict = {
        'key': 'feature_b',
        'shape': (2, 2),
        'default_value': [0.0, 0.0, 0.0, 0.0],
        'dtype': tf.float32,
        'normalizer_fn': [no_op_normalizer]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Integer dtype
    input_dict = {
        'key': 'feature_c',
        'shape': (1,),
        'default_value': [0],
        'dtype': tf.int32,
        'normalizer_fn': [no_op_normalizer]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different default value
    input_dict = {
        'key': 'feature_d',
        'shape': (1,),
        'default_value': [10.5],
        'dtype': tf.float32,
        'normalizer_fn': [no_op_normalizer]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Normalizer function
    input_dict = {
        'key': 'feature_e',
        'shape': (1,),
        'default_value': [0.0],
        'dtype': tf.float32,
        'normalizer_fn': [lambda x: x / 2.0]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: Multiple dimensions with integer type
    input_dict = {
        'key': 'feature_f',
        'shape': (3,),
        'default_value': [1, 2, 3],
        'dtype': tf.int64,
        'normalizer_fn': [no_op_normalizer]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Shape with single dimension integer
    input_dict = {
        'key': 'feature_g',
        'shape': (5,),
        'default_value': [0.1, 0.2, 0.3, 0.4, 0.5],
        'dtype': tf.float64,
        'normalizer_fn': [no_op_normalizer]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Normalizer with different operations
    input_dict = {
        'key': 'feature_h',
        'shape': (1,),
        'default_value': [5.0],
        'dtype': tf.float32,
        'normalizer_fn': [lambda x: (x - 2.0) / 3.0]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Using negative default values
    input_dict = {
        'key': 'feature_i',
        'shape': (2,),
        'default_value': [-1.0, -2.0],
        'dtype': tf.float32,
        'normalizer_fn': [no_op_normalizer]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger shape, integer and normalizer
    input_dict = {
        'key': 'feature_j',
        'shape': (2, 3),
        'default_value': [1, 2, 3, 4, 5, 6],
        'dtype': tf.int32,
        'normalizer_fn': [lambda x: x * 2]
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
