
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import os
import tempfile
import copy

def tf_data_experimental_save_inputs():
    list_of_inputs = []

    def get_temp_path(suffix):
        path = os.path.join(tempfile.gettempdir(), f"tf_data_save_test_{suffix}")
        return path

    dummy_shard_func_tensor = np.array(0, dtype=np.int64)

    # Input 1: Basic case, adhering to the flawed signature.
    input_dict_1 = {
        'dataset': np.arange(10, dtype=np.int32),
        'path': get_temp_path("1"),
        'compression': 'NONE',
        'shard_func': dummy_shard_func_tensor,
        'checkpoint_args': []
    }
    list_of_inputs.append(input_dict_1)

    # Input 2: GZIP compression.
    input_dict_2 = {
        'dataset': np.random.rand(8, 2).astype(np.float32),
        'path': get_temp_path("2"),
        'compression': 'GZIP',
        'shard_func': dummy_shard_func_tensor,
        'checkpoint_args': []
    }
    list_of_inputs.append(input_dict_2)

    # Input 3: Using a non-empty list for checkpoint_args.
    checkpoint_args_list_3 = [
        ('checkpoint_interval', 10),
        ('directory', get_temp_path("ckpt_3")),
    ]
    input_dict_3 = {
        'dataset': np.arange(50, dtype=np.int64),
        'path': get_temp_path("3"),
        'compression': 'NONE',
        'shard_func': dummy_shard_func_tensor,
        'checkpoint_args': checkpoint_args_list_3
    }
    list_of_inputs.append(input_dict_3)

    # Input 4: 2D numpy array.
    input_dict_4 = {
        'dataset': np.arange(12, dtype=np.int32).reshape(4, 3),
        'path': get_temp_path("4"),
        'compression': 'NONE',
        'shard_func': dummy_shard_func_tensor,
        'checkpoint_args': []
    }
    list_of_inputs.append(input_dict_4)

    # Input 5: Empty dataset tensor.
    input_dict_5 = {
        'dataset': np.array([], dtype=np.float32),
        'path': get_temp_path("5"),
        'compression': 'NONE',
        'shard_func': dummy_shard_func_tensor,
        'checkpoint_args': []
    }
    list_of_inputs.append(input_dict_5)

    # Input 6: Checkpointing and GZIP with a list.
    checkpoint_args_list_6 = [
        ('checkpoint_interval', 5),
        ('directory', get_temp_path("ckpt_6")),
    ]
    input_dict_6 = {
        'dataset': np.arange(20, dtype=np.int64),
        'path': get_temp_path("6"),
        'compression': 'GZIP',
        'shard_func': dummy_shard_func_tensor,
        'checkpoint_args': checkpoint_args_list_6
    }
    list_of_inputs.append(input_dict_6)

    # Input 7: 3D numpy array.
    input_dict_7 = {
        'dataset': np.random.rand(5, 2, 3).astype(np.float32),
        'path': get_temp_path("7"),
        'compression': 'NONE',
        'shard_func': dummy_shard_func_tensor,
        'checkpoint_args': []
    }
    list_of_inputs.append(input_dict_7)

    # Input 8: Negative values in dataset tensor.
    input_dict_8 = {
        'dataset': np.arange(-10, 10, dtype=np.int32),
        'path': get_temp_path("8"),
        'compression': 'GZIP',
        'shard_func': dummy_shard_func_tensor,
        'checkpoint_args': []
    }
    list_of_inputs.append(input_dict_8)

    # Input 9: Unsigned integer type.
    input_dict_9 = {
        'dataset': np.arange(15, dtype=np.uint8),
        'path': get_temp_path("9"),
        'compression': 'NONE',
        'shard_func': dummy_shard_func_tensor,
        'checkpoint_args': []
    }
    list_of_inputs.append(input_dict_9)

    # Input 10: Float16 type.
    input_dict_10 = {
        'dataset': np.random.rand(5).astype(np.float16),
        'path': get_temp_path("10"),
        'compression': 'GZIP',
        'shard_func': dummy_shard_func_tensor,
        'checkpoint_args': []
    }
    list_of_inputs.append(input_dict_10)

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
