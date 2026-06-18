
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool_with_argmax_inputs():
    list_of_inputs = []
    
    # Case 1
    input_dict = {
        'input': np.random.randn(1, 4, 4, 1).astype(np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'output_dtype': np.int64,
        'include_batch_in_index': False,
        'name': 'max_pool_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2
    input_dict = {
        'input': np.random.randint(-10, 10, size=(2, 3, 3, 2)).astype(np.int32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'output_dtype': np.int32,
        'include_batch_in_index': True,
        'name': 'max_pool_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3
    input_dict = {
        'input': np.random.randn(1, 8, 8, 3).astype(np.float64),
        'ksize': [1, 3, 3, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'output_dtype': np.int64,
        'include_batch_in_index': True,
        'name': 'max_pool_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4
    input_dict = {
        'input': np.random.randn(3, 5, 5, 4).astype(np.float16),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'output_dtype': np.int32,
        'include_batch_in_index': False,
        'name': 'max_pool_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5
    input_dict = {
        'input': np.random.randint(-5, 5, size=(1, 2, 2, 1)).astype(np.int16),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'output_dtype': np.int64,
        'include_batch_in_index': False,
        'name': 'max_pool_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6
    input_dict = {
        'input': np.random.randn(4, 10, 10, 3).astype(np.float32),
        'ksize': [1, 4, 4, 1],
        'strides': [1, 3, 3, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'output_dtype': np.int32,
        'include_batch_in_index': True,
        'name': 'max_pool_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7
    input_dict = {
        'input': np.random.randint(-128, 127, size=(2, 6, 6, 2)).astype(np.int8),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'output_dtype': np.int64,
        'include_batch_in_index': False,
        'name': 'max_pool_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8
    input_dict = {
        'input': np.random.randint(0, 255, size=(1, 7, 7, 1)).astype(np.uint8),
        'ksize': [1, 3, 3, 1],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'output_dtype': np.int32,
        'include_batch_in_index': True,
        'name': 'max_pool_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9
    input_dict = {
        'input': np.zeros((5, 4, 4, 8)).astype(np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'output_dtype': np.int64,
        'include_batch_in_index': True,
        'name': 'max_pool_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10
    input_dict = {
        'input': np.random.randint(-1000, 1000, size=(2, 2, 2, 2)).astype(np.int64),
        'ksize': [1, 1, 1, 1],
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'output_dtype': np.int32,
        'include_batch_in_index': False,
        'name': 'max_pool_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.max_pool_with_argmax"] = tf_nn_max_pool_with_argmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.max_pool_with_argmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.max_pool_with_argmax'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.max_pool_with_argmax', generated_inputs['tf.nn.max_pool_with_argmax'], lib="tf", suffix=0)
