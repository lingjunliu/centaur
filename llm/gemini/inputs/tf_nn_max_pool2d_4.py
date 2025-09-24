
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_nn_max_pool2d_inputs():
    """
    Generates a list of valid inputs for the tf.nn.max_pool2d function.
    """
    list_of_inputs = []

    # Input 1: Basic case with NHWC format
    input_1 = np.arange(1, 17, dtype=np.float32).reshape(1, 4, 4, 1)
    list_of_inputs.append(
        {
            "input": input_1,
            "ksize": [1, 2, 2, 1],
            "strides": [1, 2, 2, 1],
            "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
            "data_format": "NHWC",
            "name": "basic_nhwc"
        }
    )

    # Input 2: Another basic NHWC case (converted from NCHW to fix CPU-only error)
    input_2 = np.arange(1, 17, dtype=np.float32).reshape(1, 4, 4, 1)
    list_of_inputs.append(
        {
            "input": input_2,
            "ksize": [1, 2, 2, 1],
            "strides": [1, 2, 2, 1],
            "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
            "data_format": "NHWC",
            "name": "basic_nhwc_2"
        }
    )

    # Input 3: Explicit padding with NHWC
    input_3 = np.arange(1, 10, dtype=np.float64).reshape(1, 3, 3, 1)
    list_of_inputs.append(
        {
            "input": input_3,
            "ksize": [1, 2, 2, 1],
            "strides": [1, 2, 2, 1],
            "padding": [[0, 0], [1, 1], [1, 1], [0, 0]],
            "data_format": "NHWC",
            "name": "padding_nhwc"
        }
    )
    
    # Input 4: Explicit padding with NHWC (converted from NCHW)
    input_4 = np.arange(1, 10, dtype=np.float64).reshape(1, 3, 3, 1)
    list_of_inputs.append(
        {
            "input": input_4,
            "ksize": [1, 2, 2, 1],
            "strides": [1, 1, 1, 1],
            "padding": [[0, 0], [1, 1], [1, 1], [0, 0]],
            "data_format": "NHWC",
            "name": "padding_nhwc_2"
        }
    )

    # Input 5: Larger batch and channels, different dtype (int32)
    input_5 = np.arange(1, 151, dtype=np.int32).reshape(2, 5, 5, 3)
    list_of_inputs.append(
        {
            "input": input_5,
            "ksize": [1, 3, 3, 1],
            "strides": [1, 1, 1, 1],
            "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
            "data_format": "NHWC",
            "name": "multi_channel_batch"
        }
    )

    # Input 6: Non-square kernel and strides
    input_6 = np.arange(1, 49, dtype=np.float32).reshape(1, 6, 8, 1)
    list_of_inputs.append(
        {
            "input": input_6,
            "ksize": [1, 3, 2, 1],
            "strides": [1, 2, 3, 1],
            "padding": [[0, 0], [1, 0], [0, 1], [0, 0]],
            "data_format": "NHWC",
            "name": "nonsquare_kernel_stride"
        }
    )
    
    # Input 7: Length-2 list for ksize and strides (H, W dimensions)
    input_7 = np.arange(1, 26, dtype=np.float32).reshape(1, 5, 5, 1)
    list_of_inputs.append(
        {
            "input": input_7,
            "ksize": [2, 3],
            "strides": [2, 2],
            "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
            "data_format": "NHWC",
            "name": "hw_only_ksize_strides"
        }
    )
    
    # Input 8: Length-4 list for ksize and strides
    input_8 = np.arange(1, 37, dtype=np.float32).reshape(1, 6, 6, 1)
    list_of_inputs.append(
        {
            "input": input_8,
            "ksize": [1, 2, 2, 1],
            "strides": [1, 2, 2, 1],
            "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
            "data_format": "NHWC",
            "name": "full_ksize_strides"
        }
    )

    # Input 9: Strides > ksize
    input_9 = np.arange(1, 37, dtype=np.float32).reshape(1, 6, 6, 1)
    list_of_inputs.append(
        {
            "input": input_9,
            "ksize": [1, 2, 2, 1],
            "strides": [1, 3, 3, 1],
            "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
            "data_format": "NHWC",
            "name": "stride_gt_ksize"
        }
    )

    # Input 10: Larger valid padding
    input_10 = np.arange(1, 17, dtype=np.float32).reshape(1, 4, 4, 1)
    list_of_inputs.append(
        {
            "input": input_10,
            "ksize": [1, 3, 3, 1],
            "strides": [1, 1, 1, 1],
            "padding": [[0, 0], [2, 2], [2, 2], [0, 0]],
            "data_format": "NHWC",
            "name": "large_padding"
        }
    )
    
    # Input 11: Another dtype (float16)
    input_11 = np.arange(1, 17, dtype=np.float16).reshape(1, 4, 4, 1)
    list_of_inputs.append(
        {
            "input": input_11,
            "ksize": [1, 2, 2, 1],
            "strides": [1, 2, 2, 1],
            "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
            "data_format": "NHWC",
            "name": "float16_input"
        }
    )

    return list_of_inputs

generated_inputs["tf.nn.max_pool2d_4"] = get_tf_nn_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.max_pool2d_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.max_pool2d_4'.")

check_valid('tf.nn.max_pool2d', generated_inputs['tf.nn.max_pool2d_4'], lib="tf", suffix=4)
