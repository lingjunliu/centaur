
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MaxPool_inputs():
    list_of_inputs = []

    # Input 1, float32, NHWC, VALID
    list_of_inputs.append({
        'input': np.random.randn(1, 4, 4, 1).astype(np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "VALID",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'name': "maxpool_1"
    })

    # Input 2, float64, NHWC, SAME
    list_of_inputs.append({
        'input': np.random.randn(2, 8, 8, 3).astype(np.float64),
        'ksize': [1, 3, 3, 1],
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'name': "maxpool_2"
    })

    # Input 3, float32, NHWC, VALID
    list_of_inputs.append({
        'input': np.random.randn(1, 10, 10, 2).astype(np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "VALID",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'name': "maxpool_3"
    })

    # Input 4, int32, NHWC, SAME
    list_of_inputs.append({
        'input': np.random.randint(-50, 50, (1, 6, 6, 2)).astype(np.int32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "SAME",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'name': "maxpool_4"
    })

    # Input 5, int32, NHWC, VALID
    list_of_inputs.append({
        'input': np.random.randint(0, 255, (3, 16, 16, 4)).astype(np.int32),
        'ksize': [1, 4, 4, 1],
        'strides': [1, 4, 4, 1],
        'padding': "VALID",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'name': "maxpool_5"
    })

    # Input 6, int32, NHWC, SAME
    list_of_inputs.append({
        'input': np.random.randint(-100, 100, (1, 5, 5, 3)).astype(np.int32),
        'ksize': [1, 3, 3, 1],
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'name': "maxpool_6"
    })

    # Input 7, int32, NHWC, EXPLICIT
    list_of_inputs.append({
        'input': np.random.randint(-10, 10, (1, 4, 4, 1)).astype(np.int32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "EXPLICIT",
        'explicit_paddings': [0, 0, 1, 1, 1, 1, 0, 0],
        'data_format': "NHWC",
        'name': "maxpool_7"
    })

    # Input 8, float32, NHWC, EXPLICIT
    list_of_inputs.append({
        'input': np.random.randn(1, 14, 14, 1).astype(np.float32),
        'ksize': [1, 3, 3, 1],
        'strides': [1, 2, 2, 1],
        'padding': "EXPLICIT",
        'explicit_paddings': [0, 0, 1, 1, 2, 2, 0, 0],
        'data_format': "NHWC",
        'name': "maxpool_8"
    })

    # Input 9, int32, NHWC, SAME
    list_of_inputs.append({
        'input': np.random.randint(0, 1000, (2, 32, 32, 1)).astype(np.int32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "SAME",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'name': "maxpool_9"
    })

    # Input 10, float64, NHWC, VALID
    list_of_inputs.append({
        'input': np.random.randn(4, 8, 8, 3).astype(np.float64),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "VALID",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'name': "maxpool_10"
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPool"] = tf_raw_ops_MaxPool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MaxPool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPool'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.MaxPool', generated_inputs['tf.raw_ops.MaxPool'], lib="tf", suffix=0)
