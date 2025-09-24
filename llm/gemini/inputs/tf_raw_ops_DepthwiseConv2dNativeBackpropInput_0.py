
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def get_depthwiseconv2dnativebackpropinput_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.DepthwiseConv2dNativeBackpropInput op.
    All inputs use NHWC data format and standard dilations to be compatible with CPU execution.
    bfloat16 is excluded due to compatibility issues with the execution environment.
    """
    list_of_inputs = []

    # Helper to create inputs
    def create_input_dict(input_shape, filter_shape, out_backprop_shape, dtype, strides, padding, data_format, dilations, explicit_paddings, name):
        return {
            'input_sizes': np.array(input_shape, dtype=np.int32),
            'filter': np.random.randn(*filter_shape).astype(dtype),
            'out_backprop': np.random.randn(*out_backprop_shape).astype(dtype),
            'strides': strides,
            'padding': padding,
            'explicit_paddings': explicit_paddings,
            'data_format': data_format,
            'dilations': dilations,
            'name': name
        }

    # Case 1: Basic NHWC, VALID padding, float32
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_shape=[1, 5, 5, 3],
        filter_shape=[3, 3, 3, 2],
        out_backprop_shape=[1, 3, 3, 6],
        dtype=np.float32,
        strides=[1, 1, 1, 1],
        padding="VALID",
        data_format="NHWC",
        dilations=[1, 1, 1, 1],
        explicit_paddings=[],
        name="case1"
    )))

    # Case 2: Basic NHWC, SAME padding, float64
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_shape=[2, 8, 8, 4],
        filter_shape=[3, 3, 4, 1],
        out_backprop_shape=[2, 8, 8, 4],
        dtype=np.float64,
        strides=[1, 1, 1, 1],
        padding="SAME",
        data_format="NHWC",
        dilations=[1, 1, 1, 1],
        explicit_paddings=[],
        name="case2"
    )))

    # Case 3: NHWC, Strides > 1, VALID padding, float16 (half)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_shape=[1, 7, 7, 2],
        filter_shape=[3, 3, 2, 3],
        out_backprop_shape=[1, 3, 3, 6],
        dtype=np.float16,
        strides=[1, 2, 2, 1],
        padding="VALID",
        data_format="NHWC",
        dilations=[1, 1, 1, 1],
        explicit_paddings=[],
        name="case3"
    )))

    # Case 4: NHWC, Strides > 1, SAME padding
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_shape=[1, 7, 7, 1],
        filter_shape=[3, 3, 1, 4],
        out_backprop_shape=[1, 4, 4, 4],
        dtype=np.float32,
        strides=[1, 2, 2, 1],
        padding="SAME",
        data_format="NHWC",
        dilations=[1, 1, 1, 1],
        explicit_paddings=[],
        name="case4"
    )))

    # Case 5: NHWC, VALID padding, no dilations
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_shape=[1, 8, 8, 3],
        filter_shape=[3, 3, 3, 1],
        out_backprop_shape=[1, 6, 6, 3],
        dtype=np.float32,
        strides=[1, 1, 1, 1],
        padding="VALID",
        data_format="NHWC",
        dilations=[1, 1, 1, 1],
        explicit_paddings=[],
        name="case5"
    )))

    # Case 6: NHWC, SAME padding, float16, no dilations
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_shape=[1, 5, 5, 2],
        filter_shape=[2, 3, 2, 2],
        out_backprop_shape=[1, 5, 5, 4],
        dtype=np.float16,
        strides=[1, 1, 1, 1],
        padding="SAME",
        data_format="NHWC",
        dilations=[1, 1, 1, 1],
        explicit_paddings=[],
        name="case6"
    )))

    # Case 7: EXPLICIT padding, NHWC
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_shape=[1, 5, 5, 3],
        filter_shape=[3, 3, 3, 2],
        out_backprop_shape=[1, 5, 5, 6],
        dtype=np.float64,
        strides=[1, 1, 1, 1],
        padding="EXPLICIT",
        data_format="NHWC",
        dilations=[1, 1, 1, 1],
        explicit_paddings=[0, 0, 1, 1, 1, 1, 0, 0],
        name="case7"
    )))
    
    # Case 8: Large batch, strides > 1
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_shape=[4, 16, 16, 3],
        filter_shape=[3, 3, 3, 2],
        out_backprop_shape=[4, 8, 8, 6],
        dtype=np.float32,
        strides=[1, 2, 2, 1],
        padding="SAME",
        data_format="NHWC",
        dilations=[1, 1, 1, 1],
        explicit_paddings=[],
        name="case8"
    )))
    
    # Case 9: Asymmetric filter, no dilations
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_shape=[1, 10, 10, 4],
        filter_shape=[2, 3, 4, 1],
        out_backprop_shape=[1, 9, 8, 4],
        dtype=np.float64,
        strides=[1, 1, 1, 1],
        padding="VALID",
        data_format="NHWC",
        dilations=[1, 1, 1, 1],
        explicit_paddings=[],
        name="case9"
    )))

    # Case 10: EXPLICIT padding with strides
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_shape=[1, 7, 7, 2],
        filter_shape=[3, 3, 2, 3],
        out_backprop_shape=[1, 4, 4, 6],
        dtype=np.float32,
        strides=[1, 2, 2, 1],
        padding="EXPLICIT",
        data_format="NHWC",
        dilations=[1, 1, 1, 1],
        explicit_paddings=[0, 0, 1, 1, 1, 1, 0, 0],
        name="case10"
    )))

    # Case 11: 1x1 filter
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_shape=[2, 5, 5, 8],
        filter_shape=[1, 1, 8, 4],
        out_backprop_shape=[2, 5, 5, 32],
        dtype=np.float32,
        strides=[1, 1, 1, 1],
        padding="SAME",
        data_format="NHWC",
        dilations=[1, 1, 1, 1],
        explicit_paddings=[],
        name="case11"
    )))
    
    # Case 12: depthwise_multiplier > 1, non-square filter
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_shape=[1, 6, 8, 4],
        filter_shape=[2, 3, 4, 3],
        out_backprop_shape=[1, 5, 6, 12],
        dtype=np.float64,
        strides=[1, 1, 1, 1],
        padding="VALID",
        data_format="NHWC",
        dilations=[1, 1, 1, 1],
        explicit_paddings=[],
        name="case12"
    )))

    return list_of_inputs

generated_inputs["tf.raw_ops.DepthwiseConv2dNativeBackpropInput"] = get_depthwiseconv2dnativebackpropinput_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DepthwiseConv2dNativeBackpropInput' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DepthwiseConv2dNativeBackpropInput'.")

check_valid('tf.raw_ops.DepthwiseConv2dNativeBackpropInput', generated_inputs['tf.raw_ops.DepthwiseConv2dNativeBackpropInput'], lib="tf", suffix=0)
