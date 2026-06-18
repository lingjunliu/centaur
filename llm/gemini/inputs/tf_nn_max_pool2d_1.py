
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool2d_inputs():
    list_of_inputs = []

    # Input 1: standard NHWC, float32, positive values
    input_val = np.random.uniform(0.0, 10.0, size=(1, 4, 4, 1)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'ksize': 2,
        'strides': 2,
        'padding': "VALID",
        'data_format': "NHWC",
        'name': "pool_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: NHWC, float32, with negative values, SAME padding
    input_val = np.random.uniform(-5.0, 5.0, size=(2, 8, 8, 3)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'ksize': 3,
        'strides': 1,
        'padding': "SAME",
        'data_format': "NHWC",
        'name': "pool_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NHWC, float32, VALID padding, shape (1, 4, 4, 2)
    input_val = np.random.uniform(0.0, 1.0, size=(1, 4, 4, 2)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'ksize': 2,
        'strides': 2,
        'padding': "VALID",
        'data_format': "NHWC",
        'name': "pool_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large ksize and strides, float32, NHWC
    input_val = np.random.uniform(-10.0, 10.0, size=(1, 16, 16, 2)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'ksize': 4,
        'strides': 4,
        'padding': "SAME",
        'data_format': "NHWC",
        'name': "pool_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 data type, NHWC, VALID
    input_val = np.random.uniform(0.0, 100.0, size=(1, 6, 6, 1)).astype(np.float64)
    input_dict = {
        'input': input_val,
        'ksize': 2,
        'strides': 1,
        'padding': "VALID",
        'data_format': "NHWC",
        'name': "pool_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, NHWC, SAME
    input_val = np.random.uniform(-1.0, 1.0, size=(2, 8, 8, 3)).astype(np.float64)
    input_dict = {
        'input': input_val,
        'ksize': 2,
        'strides': 2,
        'padding': "SAME",
        'data_format': "NHWC",
        'name': "pool_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Smallest spatial size (2x2), ksize=2, strides=1, NHWC
    input_val = np.random.uniform(0.0, 5.0, size=(1, 2, 2, 1)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'ksize': 2,
        'strides': 1,
        'padding': "VALID",
        'data_format': "NHWC",
        'name': "pool_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: ksize=1, strides=1, NHWC (identity pool)
    input_val = np.random.uniform(-10.0, 10.0, size=(1, 5, 5, 4)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'ksize': 1,
        'strides': 1,
        'padding': "SAME",
        'data_format': "NHWC",
        'name': "pool_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large batch size, NHWC, float32, VALID
    input_val = np.random.uniform(0.0, 1.0, size=(8, 10, 10, 3)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'ksize': 3,
        'strides': 2,
        'padding': "VALID",
        'data_format': "NHWC",
        'name': "pool_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High channel count, float32, NHWC, SAME
    input_val = np.random.uniform(-0.5, 0.5, size=(1, 6, 6, 16)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'ksize': 2,
        'strides': 2,
        'padding': "SAME",
        'data_format': "NHWC",
        'name': "pool_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.max_pool2d_1"] = tf_nn_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.max_pool2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.max_pool2d_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.max_pool2d', generated_inputs['tf.nn.max_pool2d_1'], lib="tf", suffix=1)
