
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_AvgPool3D_inputs():
    list_of_inputs = []
    
    # 1. NDHWC, float32, SAME
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool1',
        'input': np.random.randn(1, 2, 2, 2, 1).astype(np.float32),
        'ksize': [1, 2, 2, 2, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2. NDHWC, float32, VALID
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool2',
        'input': np.random.randn(2, 4, 4, 4, 3).astype(np.float32),
        'ksize': [1, 2, 2, 2, 1],
        'strides': [1, 2, 2, 2, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3. NDHWC, float32, SAME, unit kernel
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool3',
        'input': np.random.randn(1, 3, 3, 3, 2).astype(np.float32),
        'ksize': [1, 1, 1, 1, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4. NDHWC, float32, VALID, larger kernel
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool4',
        'input': np.random.randn(2, 5, 5, 5, 1).astype(np.float32),
        'ksize': [1, 3, 3, 3, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5. NDHWC, float32, SAME, non-uniform strides
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool5',
        'input': np.random.randn(1, 4, 4, 4, 2).astype(np.float32),
        'ksize': [1, 2, 2, 2, 1],
        'strides': [1, 1, 2, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6. NDHWC, float32, VALID, non-uniform kernel
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool6',
        'input': np.random.randn(3, 3, 3, 3, 3).astype(np.float32),
        'ksize': [1, 2, 1, 2, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7. NDHWC, float32, SAME, larger kernel and strides
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool7',
        'input': np.random.randn(2, 5, 5, 5, 2).astype(np.float32),
        'ksize': [1, 3, 3, 3, 1],
        'strides': [1, 2, 2, 2, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8. NDHWC, float32, VALID, multiple channels
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool8',
        'input': np.random.randn(1, 6, 6, 6, 4).astype(np.float32),
        'ksize': [1, 2, 2, 2, 1],
        'strides': [1, 2, 2, 2, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9. NDHWC, float32, VALID, minimal input shape
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool9',
        'input': np.random.randn(1, 1, 1, 1, 1).astype(np.float32),
        'ksize': [1, 1, 1, 1, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10. NDHWC, float32, SAME, asymmetric dimensions
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool10',
        'input': np.random.randn(2, 2, 3, 4, 2).astype(np.float32),
        'ksize': [1, 2, 1, 2, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.AvgPool3D"] = tf_raw_ops_AvgPool3D_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AvgPool3D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AvgPool3D'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.AvgPool3D', generated_inputs['tf.raw_ops.AvgPool3D'], lib="tf", suffix=0)
