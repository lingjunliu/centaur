
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool1d_inputs():
    list_of_inputs = []

    # Case 1: Standard NWC, float32, VALID padding, ksize=2, stride=1
    input_dict = {
        'input': np.random.randn(2, 10, 3).astype(np.float32),
        'ksize': [2],
        'strides': [1],
        'padding': "VALID",
        'data_format': "NWC",
        'name': "pool_case_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: NWC, float32, SAME padding, ksize=3, stride=2, negative values
    input_dict = {
        'input': np.random.uniform(-5.0, -1.0, (1, 8, 2)).astype(np.float32),
        'ksize': [3],
        'strides': [2],
        'padding': "SAME",
        'data_format': "NWC",
        'name': "pool_case_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: NWC, float32, VALID padding, ksize=[1, 2, 1]
    input_dict = {
        'input': np.random.randn(4, 8, 2).astype(np.float32),
        'ksize': [1, 2, 1],
        'strides': [1, 1, 1],
        'padding': "VALID",
        'data_format': "NWC",
        'name': "pool_case_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: NWC, float64, SAME padding, ksize=2, strides=2
    input_dict = {
        'input': np.random.randn(2, 16, 4).astype(np.float64),
        'ksize': [2],
        'strides': [2],
        'padding': "SAME",
        'data_format': "NWC",
        'name': "pool_case_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: NWC, positive integers, ksize=[1, 2, 1]
    input_dict = {
        'input': np.arange(24).reshape(2, 4, 3).astype(np.float32),
        'ksize': [1, 2, 1],
        'strides': [1, 1, 1],
        'padding': "VALID",
        'data_format': "NWC",
        'name': "pool_case_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: NWC, large window size, SAME padding
    input_dict = {
        'input': np.random.uniform(-10, 10, (1, 12, 4)).astype(np.float32),
        'ksize': [4],
        'strides': [3],
        'padding': "SAME",
        'data_format': "NWC",
        'name': "pool_case_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: NWC, mixed positive and negative, ksize=[1, 3, 1], strides=[1, 2, 1]
    input_dict = {
        'input': np.random.randn(3, 10, 3).astype(np.float32),
        'ksize': [1, 3, 1],
        'strides': [1, 2, 1],
        'padding': "VALID",
        'data_format': "NWC",
        'name': "pool_case_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: NWC, zeros input, SAME padding
    input_dict = {
        'input': np.zeros((2, 8, 2), dtype=np.float32),
        'ksize': [2],
        'strides': [1],
        'padding': "SAME",
        'data_format': "NWC",
        'name': "pool_case_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: NWC, ones input, VALID padding
    input_dict = {
        'input': np.ones((1, 5, 5), dtype=np.float32),
        'ksize': [3],
        'strides': [1],
        'padding': "VALID",
        'data_format': "NWC",
        'name': "pool_case_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: NWC, float32, SAME padding, ksize=[1, 3, 1], strides=[1, 2, 1]
    input_dict = {
        'input': np.random.randn(2, 6, 4).astype(np.float32),
        'ksize': [1, 3, 1],
        'strides': [1, 2, 1],
        'padding': "SAME",
        'data_format': "NWC",
        'name': "pool_case_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.max_pool1d"] = tf_nn_max_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.max_pool1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.max_pool1d'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.max_pool1d', generated_inputs['tf.nn.max_pool1d'], lib="tf", suffix=0)
