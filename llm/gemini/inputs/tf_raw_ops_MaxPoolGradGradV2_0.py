
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MaxPoolGradGradV2_inputs():
    list_of_inputs = []
    
    # Case 1: Standard float32 NHWC
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_1',
        'orig_input': np.ones((1, 4, 4, 1), dtype=np.float32),
        'orig_output': np.ones((1, 2, 2, 1), dtype=np.float32),
        'grad': np.ones((1, 4, 4, 1), dtype=np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Case 2: float64 NHWC
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_2',
        'orig_input': np.zeros((1, 4, 4, 1), dtype=np.float64),
        'orig_output': np.zeros((1, 2, 2, 1), dtype=np.float64),
        'grad': np.zeros((1, 4, 4, 1), dtype=np.float64),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Case 3: SAME padding
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_3',
        'orig_input': np.ones((1, 4, 4, 1), dtype=np.float32),
        'orig_output': np.ones((1, 2, 2, 1), dtype=np.float32),
        'grad': np.ones((1, 4, 4, 1), dtype=np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'SAME'
    })

    # Case 4: Larger spatial dimensions NHWC
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_4',
        'orig_input': np.ones((1, 8, 8, 1), dtype=np.float32),
        'orig_output': np.ones((1, 4, 4, 1), dtype=np.float32),
        'grad': np.ones((1, 8, 8, 1), dtype=np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Case 5: Large shape, batch and channels NHWC
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_5',
        'orig_input': np.ones((2, 6, 6, 3), dtype=np.float32),
        'orig_output': np.ones((2, 2, 2, 3), dtype=np.float32),
        'grad': np.ones((2, 6, 6, 3), dtype=np.float32),
        'ksize': np.array([1, 3, 3, 1], dtype=np.int32),
        'strides': np.array([1, 3, 3, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Case 6: Different random values, float32 NHWC
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_6',
        'orig_input': np.random.rand(2, 6, 6, 3).astype(np.float32),
        'orig_output': np.random.rand(2, 2, 2, 3).astype(np.float32),
        'grad': np.random.rand(2, 6, 6, 3).astype(np.float32),
        'ksize': np.array([1, 3, 3, 1], dtype=np.int32),
        'strides': np.array([1, 3, 3, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Case 7: Non-symmetric strides and kernel sizes NHWC
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_7',
        'orig_input': np.ones((1, 5, 4, 1), dtype=np.float32),
        'orig_output': np.ones((1, 2, 3, 1), dtype=np.float32),
        'grad': np.ones((1, 5, 4, 1), dtype=np.float32),
        'ksize': np.array([1, 3, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 1, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Case 8: Negative values in inputs and grad NHWC
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_8',
        'orig_input': np.array([[[[-1.0], [2.0]], [[-3.0], [4.0]]]], dtype=np.float32),
        'orig_output': np.array([[[[4.0]]]], dtype=np.float32),
        'grad': np.array([[[[-0.5], [1.5]], [[-2.5], [3.5]]]], dtype=np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 1, 1, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Case 9: float64 type with multiple channels NHWC
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_9',
        'orig_input': np.ones((2, 4, 4, 2), dtype=np.float64),
        'orig_output': np.ones((2, 2, 2, 2), dtype=np.float64),
        'grad': np.ones((2, 4, 4, 2), dtype=np.float64),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Case 10: NHWC format with larger dimensions and 3 channels
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_10',
        'orig_input': np.ones((2, 8, 8, 3), dtype=np.float32),
        'orig_output': np.ones((2, 4, 4, 3), dtype=np.float32),
        'grad': np.ones((2, 8, 8, 3), dtype=np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPoolGradGradV2"] = tf_raw_ops_MaxPoolGradGradV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MaxPoolGradGradV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPoolGradGradV2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.MaxPoolGradGradV2', generated_inputs['tf.raw_ops.MaxPoolGradGradV2'], lib="tf", suffix=0)
