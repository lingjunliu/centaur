
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool_inputs():
    list_of_inputs = []
    
    # Input 1: 2D Spatial (NHWC), float32, SAME padding
    list_of_inputs.append({
        'input': np.random.randn(2, 4, 4, 3).astype(np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'name': "pool1"
    })
    
    # Input 2: 2D Spatial (NHWC), float32, VALID padding
    list_of_inputs.append({
        'input': np.random.randn(2, 5, 5, 3).astype(np.float32),
        'ksize': [1, 3, 3, 1],
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'name': "pool2"
    })

    # Input 3: 1D Spatial (NWC), float32, SAME padding
    list_of_inputs.append({
        'input': np.random.randn(2, 8, 3).astype(np.float32),
        'ksize': [1, 2, 1],
        'strides': [1, 2, 1],
        'padding': "SAME",
        'data_format': "NWC",
        'name': "pool3"
    })

    # Input 4: 1D Spatial (NWC), float32, VALID padding
    list_of_inputs.append({
        'input': np.random.randn(2, 8, 3).astype(np.float32),
        'ksize': [1, 3, 1],
        'strides': [1, 1, 1],
        'padding': "VALID",
        'data_format': "NWC",
        'name': "pool4"
    })

    # Input 5: 3D Spatial (NDHWC), float32, SAME padding
    list_of_inputs.append({
        'input': np.random.randn(2, 4, 4, 4, 3).astype(np.float32),
        'ksize': [1, 2, 2, 2, 1],
        'strides': [1, 2, 2, 2, 1],
        'padding': "SAME",
        'data_format': "NDHWC",
        'name': "pool5"
    })

    # Input 6: 3D Spatial (NDHWC), float32, VALID padding
    list_of_inputs.append({
        'input': np.random.randn(2, 4, 4, 4, 3).astype(np.float32),
        'ksize': [1, 3, 3, 3, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NDHWC",
        'name': "pool6"
    })

    # Input 7: 2D Spatial (NHWC) with negative values
    list_of_inputs.append({
        'input': np.array([[[[-1.0, 2.0], [3.0, -4.0]], [[5.0, -6.0], [-7.0, 8.0]]]], dtype=np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'name': "pool7"
    })

    # Input 8: 2D Spatial (NHWC) with length 1 ksize and strides
    list_of_inputs.append({
        'input': np.random.randn(2, 4, 4, 3).astype(np.float32),
        'ksize': [2],
        'strides': [2],
        'padding': "SAME",
        'data_format': "NHWC",
        'name': "pool8"
    })

    # Input 9: 2D Spatial (NHWC) with length N ksize and strides
    list_of_inputs.append({
        'input': np.random.randn(2, 6, 6, 3).astype(np.float32),
        'ksize': [3, 3],
        'strides': [1, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'name': "pool9"
    })

    # Input 10: 1D Spatial (NWC) with length 1 ksize and strides
    list_of_inputs.append({
        'input': np.random.randn(1, 10, 2).astype(np.float32),
        'ksize': [3],
        'strides': [2],
        'padding': "VALID",
        'data_format': "NWC",
        'name': "pool10"
    })

    return list_of_inputs

generated_inputs["tf.nn.max_pool"] = tf_nn_max_pool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.max_pool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.max_pool'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.max_pool', generated_inputs['tf.nn.max_pool'], lib="tf", suffix=0)
