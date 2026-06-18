
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MaxPoolGradGrad_inputs():
    list_of_inputs = []

    # Input 1: Basic NHWC float32 with VALID padding
    orig_input = np.ones((1, 4, 4, 1), dtype=np.float32)
    orig_output = np.ones((1, 2, 2, 1), dtype=np.float32)
    grad = np.random.randn(1, 4, 4, 1).astype(np.float32)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_1',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: NHWC float32 with SAME padding and kernel size 3
    orig_input = np.ones((2, 4, 4, 3), dtype=np.float32)
    orig_output = np.ones((2, 4, 4, 3), dtype=np.float32)
    grad = np.random.randn(2, 4, 4, 3).astype(np.float32)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_2',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 3, 3, 1],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NHWC float64 with VALID padding
    orig_input = np.ones((1, 6, 6, 2), dtype=np.float64)
    orig_output = np.ones((1, 3, 3, 2), dtype=np.float64)
    grad = np.random.randn(1, 6, 6, 2).astype(np.float64)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_3',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: NHWC float32 with stride 2 and SAME padding
    orig_input = np.ones((1, 5, 5, 1), dtype=np.float32)
    orig_output = np.ones((1, 3, 3, 1), dtype=np.float32)
    grad = np.random.randn(1, 5, 5, 1).astype(np.float32)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_4',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 3, 3, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: NHWC float32 with kernel size matching spatial dimensions
    orig_input = np.ones((1, 3, 3, 1), dtype=np.float32)
    orig_output = np.ones((1, 1, 1, 1), dtype=np.float32)
    grad = np.random.randn(1, 3, 3, 1).astype(np.float32)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_5',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 3, 3, 1],
        'strides': [1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: NHWC float32 with multiple channels
    orig_input = np.ones((1, 4, 4, 2), dtype=np.float32)
    orig_output = np.ones((1, 2, 2, 2), dtype=np.float32)
    grad = np.random.randn(1, 4, 4, 2).astype(np.float32)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_6',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: NHWC float32 with batch size 2
    orig_input = np.ones((2, 6, 6, 1), dtype=np.float32)
    orig_output = np.ones((2, 3, 3, 1), dtype=np.float32)
    grad = np.random.randn(2, 6, 6, 1).astype(np.float32)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_7',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: NHWC float64 with batch and multiple channels
    orig_input = np.ones((2, 2, 2, 2), dtype=np.float64)
    orig_output = np.ones((2, 1, 1, 2), dtype=np.float64)
    grad = np.random.randn(2, 2, 2, 2).astype(np.float64)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_8',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: NHWC float32 with larger input size and SAME padding
    orig_input = np.ones((1, 8, 8, 1), dtype=np.float32)
    orig_output = np.ones((1, 4, 4, 1), dtype=np.float32)
    grad = np.random.randn(1, 8, 8, 1).astype(np.float32)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_9',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: NHWC float32 with 8x8 input and multiple channels
    orig_input = np.ones((1, 8, 8, 3), dtype=np.float32)
    orig_output = np.ones((1, 4, 4, 3), dtype=np.float32)
    grad = np.random.randn(1, 8, 8, 3).astype(np.float32)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_10',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPoolGradGrad"] = tf_raw_ops_MaxPoolGradGrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MaxPoolGradGrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPoolGradGrad'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.MaxPoolGradGrad', generated_inputs['tf.raw_ops.MaxPoolGradGrad'], lib="tf", suffix=0)
