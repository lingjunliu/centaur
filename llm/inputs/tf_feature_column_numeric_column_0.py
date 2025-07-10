
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_feature_column_numeric_column_inputs():
    list_of_inputs = []

    # Input 1, valid
    key = 'feature_a'
    shape = (1,)
    default_value = [0.0]
    dtype = tf.float32
    normalizer_fn = None

    input_dict = {
        'key': key,
        'shape': shape,
        'default_value': default_value,
        'dtype': dtype,
        'normalizer_fn': normalizer_fn if normalizer_fn is not None else []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    key = 'feature_b'
    shape = (2,)
    default_value = [1.0, 2.0]
    dtype = tf.float32
    normalizer_fn = None

    input_dict = {
        'key': key,
        'shape': shape,
        'default_value': default_value,
        'dtype': dtype,
        'normalizer_fn': normalizer_fn if normalizer_fn is not None else []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    key = 'feature_c'
    shape = (1, 3)
    default_value = [1.0, 2.0, 3.0]
    dtype = tf.float32
    normalizer_fn = None

    input_dict = {
        'key': key,
        'shape': shape,
        'default_value': default_value,
        'dtype': dtype,
        'normalizer_fn': normalizer_fn if normalizer_fn is not None else []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    key = 'feature_d'
    shape = (1,)
    default_value = [-1.0]
    dtype = tf.float32
    normalizer_fn = None

    input_dict = {
        'key': key,
        'shape': shape,
        'default_value': default_value,
        'dtype': dtype,
        'normalizer_fn': normalizer_fn if normalizer_fn is not None else []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, int dtype
    key = 'feature_e'
    shape = (1,)
    default_value = [1]
    dtype = tf.int32
    normalizer_fn = None

    input_dict = {
        'key': key,
        'shape': shape,
        'default_value': default_value,
        'dtype': dtype,
        'normalizer_fn': normalizer_fn if normalizer_fn is not None else []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid, multiple dimensions, int dtype
    key = 'feature_f'
    shape = (2, 2)
    default_value = [1, 2, 3, 4]
    dtype = tf.int32
    normalizer_fn = None

    input_dict = {
        'key': key,
        'shape': shape,
        'default_value': default_value,
        'dtype': dtype,
        'normalizer_fn': normalizer_fn if normalizer_fn is not None else []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, normalizer function
    key = 'feature_g'
    shape = (1,)
    default_value = [5.0]
    dtype = tf.float32
    normalizer_fn = lambda x: (x - 3.0) / 4.2

    input_dict = {
        'key': key,
        'shape': shape,
        'default_value': default_value,
        'dtype': dtype,
        'normalizer_fn': [normalizer_fn]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid, different shape
    key = 'feature_h'
    shape = (3,)
    default_value = [1.1, 2.2, 3.3]
    dtype = tf.float32
    normalizer_fn = None

    input_dict = {
        'key': key,
        'shape': shape,
        'default_value': default_value,
        'dtype': dtype,
        'normalizer_fn': normalizer_fn if normalizer_fn is not None else []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid, large shape
    key = 'feature_i'
    shape = (2, 3, 4)
    default_value = list(range(24))
    dtype = tf.int32
    normalizer_fn = None

    input_dict = {
        'key': key,
        'shape': shape,
        'default_value': default_value,
        'dtype': dtype,
        'normalizer_fn': normalizer_fn if normalizer_fn is not None else []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid, large negative values
    key = 'feature_j'
    shape = (1,)
    default_value = [-1000.0]
    dtype = tf.float32
    normalizer_fn = None

    input_dict = {
        'key': key,
        'shape': shape,
        'default_value': default_value,
        'dtype': dtype,
        'normalizer_fn': normalizer_fn if normalizer_fn is not None else []
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
