
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_avg_pool1d_inputs():
    list_of_inputs = []
    
    # Input 1
    list_of_inputs.append({
        'input': np.random.randn(2, 10, 3).astype(np.float32),
        'ksize': 2,
        'strides': 2,
        'padding': 'VALID',
        'data_format': 'NWC',
        'name': 'pool1'
    })
    
    # Input 2
    list_of_inputs.append({
        'input': np.random.randn(1, 16, 4).astype(np.float32),
        'ksize': 3,
        'strides': 1,
        'padding': 'SAME',
        'data_format': 'NWC',
        'name': 'pool2'
    })
    
    # Input 3
    list_of_inputs.append({
        'input': np.random.randn(4, 8, 2).astype(np.float32),
        'ksize': 2,
        'strides': 1,
        'padding': 'VALID',
        'data_format': 'NWC',
        'name': 'pool3'
    })
    
    # Input 4
    list_of_inputs.append({
        'input': np.random.randn(3, 20, 3).astype(np.float64),
        'ksize': 4,
        'strides': 2,
        'padding': 'SAME',
        'data_format': 'NWC',
        'name': 'pool4'
    })
    
    # Input 5
    list_of_inputs.append({
        'input': np.arange(24).reshape(2, 4, 3).astype(np.float32),
        'ksize': 2,
        'strides': 2,
        'padding': 'VALID',
        'data_format': 'NWC',
        'name': 'pool5'
    })
    
    # Input 6
    list_of_inputs.append({
        'input': np.ones((1, 100, 1), dtype=np.float32),
        'ksize': 5,
        'strides': 5,
        'padding': 'SAME',
        'data_format': 'NWC',
        'name': 'pool6'
    })
    
    # Input 7
    list_of_inputs.append({
        'input': np.zeros((2, 10, 5), dtype=np.float32),
        'ksize': 3,
        'strides': 3,
        'padding': 'VALID',
        'data_format': 'NWC',
        'name': 'pool7'
    })
    
    # Input 8
    list_of_inputs.append({
        'input': np.random.uniform(-10, 10, (5, 8, 2)).astype(np.float32),
        'ksize': 1,
        'strides': 1,
        'padding': 'VALID',
        'data_format': 'NWC',
        'name': 'pool8'
    })
    
    # Input 9
    list_of_inputs.append({
        'input': np.random.normal(0, 1, (2, 12, 2)).astype(np.float64),
        'ksize': 3,
        'strides': 2,
        'padding': 'SAME',
        'data_format': 'NWC',
        'name': 'pool9'
    })
    
    # Input 10
    list_of_inputs.append({
        'input': np.ones((4, 30, 3), dtype=np.float32),
        'ksize': 10,
        'strides': 5,
        'padding': 'SAME',
        'data_format': 'NWC',
        'name': 'pool10'
    })
    
    return list_of_inputs

generated_inputs["tf.nn.avg_pool1d_1"] = tf_nn_avg_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.avg_pool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.avg_pool1d_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.avg_pool1d', generated_inputs['tf.nn.avg_pool1d_1'], lib="tf", suffix=1)
