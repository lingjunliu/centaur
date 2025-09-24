
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_max_pool_3d_grad_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.MaxPool3DGrad.
    """

    def _create_input_dict(orig_input_shape, ksize, strides, padding, data_format, dtype, name="test"):
        # The error indicates that the CPU implementation of MaxPool3DGrad only supports 'NDHWC'.
        # We force operations to run on CPU during input generation to simulate the test environment
        # and ensure the generated inputs are valid for it.
        with tf.device('/CPU:0'):
            total_elements = np.prod(orig_input_shape)
            if total_elements == 0:
                orig_input_np = np.zeros(orig_input_shape, dtype=dtype)
            else:
                orig_input_np = np.arange(total_elements, dtype=dtype).reshape(orig_input_shape)
                np.random.shuffle(orig_input_np.flatten())

            orig_input_tf = tf.constant(orig_input_np, dtype=tf.as_dtype(dtype))

            orig_output_tf = tf.nn.max_pool3d(
                input=orig_input_tf,
                ksize=ksize,
                strides=strides,
                padding=padding.upper(),
                data_format=data_format
            )
            orig_output_np = orig_output_tf.numpy()
        
        grad_np = np.random.randn(*orig_output_np.shape).astype(dtype)
        
        return {
            'orig_input': orig_input_np,
            'orig_output': orig_output_np,
            'grad': grad_np,
            'ksize': ksize,
            'strides': strides,
            'padding': padding.upper(),
            'data_format': data_format,
            'name': name
        }

    # Test cases are restricted to data_format='NDHWC' because the CPU
    # implementation of MaxPool3DGrad does not support 'NCDHW'.
    test_cases = [
        # Case 1: Basic NDHWC, VALID padding, float32
        {'orig_input_shape': (2, 4, 4, 4, 3), 'ksize': [1, 2, 2, 2, 1], 'strides': [1, 2, 2, 2, 1], 'padding': 'VALID', 'data_format': 'NDHWC', 'dtype': np.float32},
        # Case 2: Basic NDHWC, SAME padding, float16
        {'orig_input_shape': (1, 3, 5, 4, 2), 'ksize': [1, 3, 3, 3, 1], 'strides': [1, 1, 1, 1, 1], 'padding': 'SAME', 'data_format': 'NDHWC', 'dtype': np.float16},
        # Case 3: NDHWC, non-cubic kernel/strides, VALID padding
        {'orig_input_shape': (1, 6, 8, 10, 1), 'ksize': [1, 2, 3, 4, 1], 'strides': [1, 1, 2, 3, 1], 'padding': 'VALID', 'data_format': 'NDHWC', 'dtype': np.float32},
        # Case 4: NDHWC, non-cubic kernel/strides, SAME padding
        {'orig_input_shape': (1, 7, 9, 11, 2), 'ksize': [1, 4, 3, 2, 1], 'strides': [1, 3, 2, 1, 1], 'padding': 'SAME', 'data_format': 'NDHWC', 'dtype': np.float32},
        # Case 5: NDHWC, stride > kernel size
        {'orig_input_shape': (1, 10, 10, 10, 1), 'ksize': [1, 2, 2, 2, 1], 'strides': [1, 3, 3, 3, 1], 'padding': 'SAME', 'data_format': 'NDHWC', 'dtype': np.float32},
        # Case 6: NDHWC, ksize=1 (identity mapping for pooling)
        {'orig_input_shape': (2, 5, 5, 5, 3), 'ksize': [1, 1, 1, 1, 1], 'strides': [1, 1, 1, 1, 1], 'padding': 'VALID', 'data_format': 'NDHWC', 'dtype': np.float16},
        # Case 7: NDHWC, input with a zero-sized dimension
        {'orig_input_shape': (1, 5, 0, 5, 2), 'ksize': [1, 2, 2, 2, 1], 'strides': [1, 1, 1, 1, 1], 'padding': 'SAME', 'data_format': 'NDHWC', 'dtype': np.float32},
        # Case 8: NDHWC, larger batch and channels
        {'orig_input_shape': (4, 4, 4, 4, 8), 'ksize': [1, 2, 2, 2, 1], 'strides': [1, 1, 1, 1, 1], 'padding': 'SAME', 'data_format': 'NDHWC', 'dtype': np.float32},
        # Case 9: NDHWC, VALID padding with large strides
        {'orig_input_shape': (1, 8, 8, 8, 1), 'ksize': [1, 3, 3, 3, 1], 'strides': [1, 3, 3, 3, 1], 'padding': 'VALID', 'data_format': 'NDHWC', 'dtype': np.float16},
        # Case 10: NDHWC, larger, non-symmetrical dimensions
        {'orig_input_shape': (1, 12, 10, 8, 4), 'ksize': [1, 2, 2, 2, 1], 'strides': [1, 2, 2, 2, 1], 'padding': 'SAME', 'data_format': 'NDHWC', 'dtype': np.float32},
        # Case 11: NDHWC, another non-cubic kernel/stride combo
        {'orig_input_shape': (2, 7, 7, 7, 2), 'ksize': [1, 1, 3, 3, 1], 'strides': [1, 1, 2, 2, 1], 'padding': 'VALID', 'data_format': 'NDHWC', 'dtype': np.float32},
    ]

    list_of_inputs = []
    for i, case_params in enumerate(test_cases):
        try:
            input_dict = _create_input_dict(**case_params, name=f"test_case_{i+1}")
            list_of_inputs.append(copy.deepcopy(input_dict))
        except Exception:
            # Silently skip any problematic case during generation
            continue

    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPool3DGrad"] = tf_raw_ops_max_pool_3d_grad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MaxPool3DGrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPool3DGrad'.")

check_valid('tf.raw_ops.MaxPool3DGrad', generated_inputs['tf.raw_ops.MaxPool3DGrad'], lib="tf", suffix=0)
