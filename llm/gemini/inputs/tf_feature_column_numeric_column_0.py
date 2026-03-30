
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_feature_column_numeric_column_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    input_dict = {
        'key': 'feature_a',
        'shape': (1,),
        'default_value': [0.0],
        'dtype': tf.float32,
        'normalizer_fn': lambda x: x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shape
    input_dict = {
        'key': 'feature_b',
        'shape': (2, 2),
        'default_value': [0.0, 0.0, 0.0, 0.0],
        'dtype': tf.float32,
        'normalizer_fn': lambda x: x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Integer dtype
    input_dict = {
        'key': 'feature_c',
        'shape': (1,),
        'default_value': [0],
        'dtype': tf.int32,
        'normalizer_fn': lambda x: x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: different key
    input_dict = {
        'key': 'feature_d',
        'shape': (1,),
        'default_value': [1.0],
        'dtype': tf.float32,
        'normalizer_fn': lambda x: x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5: different default value
    input_dict = {
        'key': 'feature_e',
        'shape': (1,),
        'default_value': [-1.0],
        'dtype': tf.float32,
        'normalizer_fn': lambda x: x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: multi-dimensional shape with int dtype
    input_dict = {
        'key': 'feature_f',
        'shape': (3,),
        'default_value': [1, 2, 3],
        'dtype': tf.int64,
        'normalizer_fn': lambda x: x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: shape as int
    input_dict = {
        'key': 'feature_g',
        'shape': (1,),
        'default_value': [5.0],
        'dtype': tf.float64,
        'normalizer_fn': lambda x: x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: larger shape
    input_dict = {
        'key': 'feature_h',
        'shape': (5,),
        'default_value': [1.0, 2.0, 3.0, 4.0, 5.0],
        'dtype': tf.float32,
        'normalizer_fn': lambda x: x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: different dtype
    input_dict = {
        'key': 'feature_i',
        'shape': (1,),
        'default_value': [10],
        'dtype': tf.int16,
        'normalizer_fn': lambda x: x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: negative default value, integer type, multidimensional shape
    input_dict = {
        'key': 'feature_j',
        'shape': (2, 1),
        'default_value': [-1, -2],
        'dtype': tf.int32,
        'normalizer_fn': lambda x: x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
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


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.feature_column.numeric_column', generated_inputs['tf.feature_column.numeric_column'], lib="tf", suffix=0)
