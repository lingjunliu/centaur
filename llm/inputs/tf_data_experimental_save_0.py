
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os

def tf_data_experimental_save_inputs():
    list_of_inputs = []
    base_path = '/tmp/tf_data_experimental_save'

    # Input 1: Basic case with 1D integer data
    list_of_inputs.append(copy.deepcopy({
        'dataset': np.arange(10, dtype=np.int32),
        'path': os.path.join(base_path, '1'),
        'compression': 'NONE',
        'shard_func': np.array(0, dtype=np.int64),
        'checkpoint_args': [],
    }))
    
    # Input 2: 2D float data with GZIP compression
    list_of_inputs.append(copy.deepcopy({
        'dataset': np.random.rand(5, 3).astype(np.float32),
        'path': os.path.join(base_path, '2'),
        'compression': 'GZIP',
        'shard_func': np.array(0, dtype=np.int64),
        'checkpoint_args': [],
    }))
    
    # Input 3: Dataset of strings with a specific shard_func
    list_of_inputs.append(copy.deepcopy({
        'dataset': np.array(['alpha', 'beta', 'gamma']),
        'path': os.path.join(base_path, '3'),
        'compression': 'NONE',
        'shard_func': np.array(1, dtype=np.int64),
        'checkpoint_args': [],
    }))
    
    # Input 4: Large dataset with checkpointing
    list_of_inputs.append(copy.deepcopy({
        'dataset': np.arange(100, dtype=np.int64),
        'path': os.path.join(base_path, '4'),
        'compression': 'NONE',
        'shard_func': np.array(0, dtype=np.int64),
        'checkpoint_args': [
            ('checkpoint_interval', 10),
            ('step_counter', np.array(0, dtype=np.int64)),
            ('directory', os.path.join(base_path, 'ckpt_4')),
            ('max_to_keep', 5)
        ],
    }))
    
    # Input 5: All optional arguments used
    list_of_inputs.append(copy.deepcopy({
        'dataset': np.random.rand(10, 2).astype(np.float64),
        'path': os.path.join(base_path, '5'),
        'compression': 'GZIP',
        'shard_func': np.array(2, dtype=np.int64),
        'checkpoint_args': [
            ('checkpoint_interval', 5),
            ('step_counter', np.array(10, dtype=np.int64)),
            ('directory', os.path.join(base_path, 'ckpt_5')),
            ('max_to_keep', 2)
        ],
    }))
    
    # Input 6: 3D data with int8 type
    list_of_inputs.append(copy.deepcopy({
        'dataset': np.ones((2, 3, 4), dtype=np.int8),
        'path': os.path.join(base_path, '6'),
        'compression': 'NONE',
        'shard_func': np.array(0, dtype=np.int64),
        'checkpoint_args': [],
    }))
    
    # Input 7: Complex number dataset
    list_of_inputs.append(copy.deepcopy({
        'dataset': np.array([1+2j, 3-4j, 5+6j], dtype=np.complex64),
        'path': os.path.join(base_path, '7'),
        'compression': 'NONE',
        'shard_func': np.array(0, dtype=np.int64),
        'checkpoint_args': [],
    }))
    
    # Input 8: Empty dataset
    list_of_inputs.append(copy.deepcopy({
        'dataset': np.array([], dtype=np.float32),
        'path': os.path.join(base_path, '8'),
        'compression': 'NONE',
        'shard_func': np.array(0, dtype=np.int64),
        'checkpoint_args': [],
    }))
    
    # Input 9: Boolean dataset
    list_of_inputs.append(copy.deepcopy({
        'dataset': np.array([True, False, True, True, False]),
        'path': os.path.join(base_path, '9'),
        'compression': 'NONE',
        'shard_func': np.array(0, dtype=np.int64),
        'checkpoint_args': [],
    }))
    
    # Input 10: Single element dataset with GZIP and sharding
    list_of_inputs.append(copy.deepcopy({
        'dataset': np.array([1337], dtype=np.uint32),
        'path': os.path.join(base_path, '10'),
        'compression': 'GZIP',
        'shard_func': np.array(3, dtype=np.int64),
        'checkpoint_args': [],
    }))

    return list_of_inputs

generated_inputs["tf.data.experimental.save"] = tf_data_experimental_save_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.save' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.save'.")

check_valid('tf.data.experimental.save', generated_inputs['tf.data.experimental.save'], lib="tf", suffix=0)
