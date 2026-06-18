
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_fractional_max_pool_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        'value': np.random.randn(1, 10, 10, 1).astype(np.float32),
        'pooling_ratio': [1.0, 1.5, 1.5, 1.0],
        'pseudo_random': True,
        'overlapping': True,
        'seed': 42,
        'name': "pool1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {
        'value': np.random.randn(2, 20, 20, 3).astype(np.float32),
        'pooling_ratio': [1.0, 1.2, 1.8, 1.0],
        'pseudo_random': False,
        'overlapping': False,
        'seed': 1,
        'name': "pool2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {
        'value': np.random.randint(-10, 10, size=(1, 5, 5, 1)).astype(np.int32),
        'pooling_ratio': [1.0, 1.1, 1.1, 1.0],
        'pseudo_random': True,
        'overlapping': False,
        'seed': 10,
        'name': "pool3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {
        'value': np.random.randn(4, 15, 15, 2).astype(np.float64),
        'pooling_ratio': [1.0, 1.44, 1.73, 1.0],
        'pseudo_random': False,
        'overlapping': True,
        'seed': 100,
        'name': "pool4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        'value': np.random.randn(1, 100, 100, 3).astype(np.float32),
        'pooling_ratio': [1.0, 5.5, 5.5, 1.0],
        'pseudo_random': True,
        'overlapping': True,
        'seed': 7,
        'name': "pool5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {
        'value': np.random.randint(-50, 50, size=(3, 8, 8, 4)).astype(np.int64),
        'pooling_ratio': [1.0, 1.0, 1.0, 1.0],
        'pseudo_random': False,
        'overlapping': False,
        'seed': 88,
        'name': "pool6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {
        'value': np.random.randn(1, 50, 50, 1).astype(np.float32),
        'pooling_ratio': [1.0, 2.5, 3.5, 1.0],
        'pseudo_random': True,
        'overlapping': False,
        'seed': 999,
        'name': "pool7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {
        'value': np.random.randn(2, 12, 16, 3).astype(np.float32),
        'pooling_ratio': [1.0, 1.3, 1.4, 1.0],
        'pseudo_random': False,
        'overlapping': True,
        'seed': 12345,
        'name': "pool8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {
        'value': np.random.randn(1, 7, 9, 2).astype(np.float32),
        'pooling_ratio': [1.0, 1.05, 1.15, 1.0],
        'pseudo_random': True,
        'overlapping': True,
        'seed': 3,
        'name': "pool9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        'value': np.random.randn(5, 25, 25, 5).astype(np.float32),
        'pooling_ratio': [1.0, 2.1, 2.1, 1.0],
        'pseudo_random': False,
        'overlapping': False,
        'seed': 12,
        'name': "pool10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.fractional_max_pool"] = tf_nn_fractional_max_pool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.fractional_max_pool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.fractional_max_pool'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.fractional_max_pool', generated_inputs['tf.nn.fractional_max_pool'], lib="tf", suffix=0)
