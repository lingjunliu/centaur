
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_maxpool3dgradgrad_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.MaxPool3DGradGrad function.
    """
    
    def _create_input_pair(in_shape, ksize, strides, padding, data_format, dtype):
        """
        Helper to generate a valid (orig_input, orig_output, grad) triplet.
        It uses tf.nn.max_pool3d to ensure orig_output is a valid result of
        pooling orig_input. Runs on CPU to avoid potential cuDNN environment errors.
        """
        orig_input_np = np.random.uniform(-10, 10, size=in_shape).astype(dtype)

        if dtype == tf.bfloat16.as_numpy_dtype:
            orig_input_tf = tf.cast(tf.constant(orig_input_np.astype(np.float32)), dtype=tf.bfloat16)
        else:
            orig_input_tf = tf.constant(orig_input_np, dtype=dtype)
        
        # Force the helper operation to run on CPU to avoid GPU/cuDNN errors
        with tf.device('/CPU:0'):
            # tf.nn.max_pool3d only supports float, half, and bfloat16.
            # We generate inputs based on these supported types.
            float_input_for_pool = tf.cast(orig_input_tf, tf.float32)
            orig_output_tf_float = tf.nn.max_pool3d(
                input=float_input_for_pool,
                ksize=ksize,
                strides=strides,
                padding=padding.upper(),
                data_format=data_format
            )
            orig_output_tf = tf.cast(orig_output_tf_float, orig_input_tf.dtype)

        # The 'grad' param for MaxPool3DGradGrad is the incoming gradient to the
        # MaxPool3DGrad op, so it has the same shape as its output, which is
        # the same shape as orig_input.
        grad_shape = in_shape
        grad_np = np.random.uniform(-10, 10, size=grad_shape).astype(dtype)
        
        if dtype == tf.bfloat16.as_numpy_dtype:
            grad_tf = tf.cast(tf.constant(grad_np.astype(np.float32)), dtype=tf.bfloat16)
        else:
            grad_tf = tf.constant(grad_np, dtype=dtype)

        return orig_input_tf.numpy(), orig_output_tf.numpy(), grad_tf.numpy()

    list_of_inputs = []

    # tf.nn.max_pool3d, used for data generation, only supports float, half, and bfloat16.
    # We restrict the generated types accordingly to prevent InvalidArgumentError.
    supported_dtypes = [np.float32, np.float16, tf.bfloat16.as_numpy_dtype]

    # A list of configurations to generate diverse inputs
    configs = [
        # Case 1: float32, NDHWC, VALID padding
        {"in_shape": (1, 4, 4, 4, 1), "ksize": [1, 2, 2, 2, 1], "strides": [1, 2, 2, 2, 1], "padding": "VALID", "data_format": "NDHWC", "dtype": np.float32},
        # Case 2: float32, NDHWC, SAME padding
        {"in_shape": (1, 3, 3, 3, 2), "ksize": [1, 2, 2, 2, 1], "strides": [1, 1, 1, 1, 1], "padding": "SAME", "data_format": "NDHWC", "dtype": np.float32},
        # Case 3: float32, NCDHW, VALID padding
        {"in_shape": (1, 2, 5, 5, 5), "ksize": [1, 1, 3, 3, 3], "strides": [1, 1, 1, 1, 1], "padding": "VALID", "data_format": "NCDHW", "dtype": np.float32},
        # Case 4: float32, NCDHW, SAME padding, larger strides
        {"in_shape": (2, 1, 6, 6, 6), "ksize": [1, 1, 3, 3, 3], "strides": [1, 1, 3, 3, 3], "padding": "SAME", "data_format": "NCDHW", "dtype": np.float32},
        # Case 5: float16 (half), NDHWC, VALID padding
        {"in_shape": (1, 5, 6, 7, 3), "ksize": [1, 2, 3, 4, 1], "strides": [1, 1, 2, 2, 1], "padding": "VALID", "data_format": "NDHWC", "dtype": np.float16},
        # Case 6: float16 (half), NCDHW, SAME padding
        {"in_shape": (1, 2, 5, 5, 5), "ksize": [1, 1, 2, 2, 2], "strides": [1, 1, 2, 2, 2], "padding": "SAME", "data_format": "NCDHW", "dtype": np.float16},
        # Case 7: bfloat16, NDHWC, SAME padding
        {"in_shape": (1, 4, 4, 4, 2), "ksize": [1, 2, 2, 2, 1], "strides": [1, 1, 1, 1, 1], "padding": "SAME", "data_format": "NDHWC", "dtype": tf.bfloat16.as_numpy_dtype},
        # Case 8: bfloat16, NCDHW, VALID padding
        {"in_shape": (2, 3, 6, 6, 6), "ksize": [1, 1, 3, 3, 3], "strides": [1, 1, 2, 2, 2], "padding": "VALID", "data_format": "NCDHW", "dtype": tf.bfloat16.as_numpy_dtype},
        # Case 9: float32, large batch and channels, SAME padding
        {"in_shape": (4, 5, 5, 5, 3), "ksize": [1, 3, 3, 3, 1], "strides": [1, 2, 2, 2, 1], "padding": "SAME", "data_format": "NDHWC", "dtype": np.float32},
        # Case 10: float16, non-cubic ksize/strides, VALID padding
        {"in_shape": (1, 2, 8, 8, 8), "ksize": [1, 1, 2, 3, 4], "strides": [1, 1, 4, 3, 2], "padding": "VALID", "data_format": "NCDHW", "dtype": np.float16},
        # Case 11: float32, edge case where ksize > input dim with SAME padding
        {"in_shape": (1, 2, 2, 2, 1), "ksize": [1, 3, 3, 3, 1], "strides": [1, 1, 1, 1, 1], "padding": "SAME", "data_format": "NDHWC", "dtype": np.float32},
    ]

    for i, config in enumerate(configs):
        orig_input, orig_output, grad = _create_input_pair(
            in_shape=config["in_shape"],
            ksize=config["ksize"],
            strides=config["strides"],
            padding=config["padding"],
            data_format=config["data_format"],
            dtype=config["dtype"]
        )
        
        input_dict = {
            'orig_input': orig_input,
            'orig_output': orig_output,
            'grad': grad,
            'ksize': config["ksize"],
            'strides': config["strides"],
            'padding': config["padding"],
            'data_format': config["data_format"],
            'name': f'case_{i+1}'
        }
        list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPool3DGradGrad"] = get_maxpool3dgradgrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MaxPool3DGradGrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPool3DGradGrad'.")

check_valid('tf.raw_ops.MaxPool3DGradGrad', generated_inputs['tf.raw_ops.MaxPool3DGradGrad'], lib="tf", suffix=0)
