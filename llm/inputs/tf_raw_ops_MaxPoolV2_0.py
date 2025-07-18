
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_maxpoolv2_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.MaxPoolV2 function.
    """
    list_of_inputs = []

    # Input 1: Basic case with float32, NHWC, and VALID padding
    input_dict = {
        'input': np.random.rand(1, 4, 4, 1).astype(np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': "VALID",
        'data_format': 'NHWC',
        'name': 'test_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic case with float32, NHWC, and SAME padding
    input_dict = {
        'input': np.random.rand(1, 5, 5, 3).astype(np.float32),
        'ksize': np.array([1, 3, 3, 1], dtype=np.int32),
        'strides': np.array([1, 1, 1, 1], dtype=np.int32),
        'padding': "SAME",
        'data_format': 'NHWC',
        'name': 'test_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Corrected from NCHW to NHWC for CPU compatibility
    input_dict = {
        'input': np.random.rand(1, 6, 6, 3).astype(np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': "VALID",
        'data_format': 'NHWC',
        'name': 'test_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data type (int32)
    input_dict = {
        'input': np.arange(1 * 8 * 8 * 2).reshape(1, 8, 8, 2).astype(np.int32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': "VALID",
        'data_format': 'NHWC',
        'name': 'test_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 data type with NHWC and SAME padding (Corrected from NCHW)
    input_dict = {
        'input': np.random.rand(2, 7, 7, 2).astype(np.float64),
        'ksize': np.array([1, 3, 3, 1], dtype=np.int32),
        'strides': np.array([1, 3, 3, 1], dtype=np.int32),
        'padding': "SAME",
        'data_format': 'NHWC',
        'name': 'test_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Non-square kernel and strides
    input_dict = {
        'input': np.random.rand(1, 10, 8, 1).astype(np.float32),
        'ksize': np.array([1, 4, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 1, 1], dtype=np.int32),
        'padding': "VALID",
        'data_format': 'NHWC',
        'name': 'test_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint8 data type
    input_dict = {
        'input': (np.random.rand(1, 16, 16, 3) * 255).astype(np.uint8),
        'ksize': np.array([1, 4, 4, 1], dtype=np.int32),
        'strides': np.array([1, 4, 4, 1], dtype=np.int32),
        'padding': "VALID",
        'data_format': 'NHWC',
        'name': 'test_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half (float16) data type
    input_dict = {
        'input': np.random.rand(1, 8, 8, 1).astype(np.float16),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 1, 1, 1], dtype=np.int32),
        'padding': "SAME",
        'data_format': 'NHWC',
        'name': 'test_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Strides larger than kernel size
    input_dict = {
        'input': np.arange(1 * 12 * 12 * 1).reshape(1, 12, 12, 1).astype(np.int64),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 3, 3, 1], dtype=np.int32),
        'padding': "VALID",
        'data_format': 'NHWC',
        'name': 'test_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger batch size and channels with NHWC (Corrected from NCHW)
    input_dict = {
        'input': np.random.rand(4, 16, 16, 8).astype(np.float32),
        'ksize': np.array([1, 3, 3, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': "SAME",
        'data_format': 'NHWC',
        'name': 'test_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: int8 data type
    input_dict = {
        'input': (np.random.rand(1, 5, 5, 1) * 254 - 127).astype(np.int8),
        'ksize': np.array([1, 3, 3, 1], dtype=np.int32),
        'strides': np.array([1, 1, 1, 1], dtype=np.int32),
        'padding': "VALID",
        'data_format': 'NHWC',
        'name': 'test_11'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPoolV2"] = get_tf_raw_ops_maxpoolv2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MaxPoolV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPoolV2'.")

check_valid('tf.raw_ops.MaxPoolV2', generated_inputs['tf.raw_ops.MaxPoolV2'], lib="tf", suffix=0)
