
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv_transpose_inputs():
    list_of_inputs = []

    # Input 3: Matching depths, adjusted output shape calculation
    input3 = np.random.rand(1, 3, 3, 1).astype(np.float32)
    filters3 = np.random.rand(2, 2, 1, 1).astype(np.float32)
    output_shape3 = np.array([1, 4, 4, 1]).astype(np.int32) # Adjusted for SAME padding
    strides3 = 1
    padding3 = 'SAME'
    data_format3 = 'NHWC'
    dilations3 = 1
    name3 = 'conv_transpose3'

    input_dict3 = {
        "input": input3,
        "filters": filters3,
        "output_shape": output_shape3,
        "strides": strides3,
        "padding": padding3,
        "data_format": data_format3,
        "dilations": dilations3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 10, with name None, matching depths
    input10 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filters10 = np.random.rand(3, 3, 3, 3).astype(np.float32)
    output_shape10 = np.array([1, 7, 7, 3]).astype(np.int32)
    strides10 = 1
    padding10 = 'SAME'
    data_format10 = 'NHWC'
    dilations10 = 1
    name10 = None

    input_dict10 = {
        "input": input10,
        "filters": filters10,
        "output_shape": output_shape10,
        "strides": strides10,
        "padding": padding10,
        "data_format": data_format10,
        "dilations": dilations10,
        "name": name10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11: Matching depths
    input11 = np.random.rand(1, 5, 5, 2).astype(np.float32)
    filters11 = np.random.rand(3, 3, 2, 4).astype(np.float32)
    output_shape11 = np.array([1, 7, 7, 4]).astype(np.int32)
    strides11 = 1
    padding11 = 'SAME'
    data_format11 = 'NHWC'
    dilations11 = 1
    name11 = "conv_transpose11"

    input_dict11 = {
        "input": input11,
        "filters": filters11,
        "output_shape": output_shape11,
        "strides": strides11,
        "padding": padding11,
        "data_format": data_format11,
        "dilations": dilations11,
        "name": name11
    }
    list_of_inputs.append(copy.deepcopy(input_dict11))

    # Input 12: VALID padding, adjusted output shape, matching depths
    input12 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filters12 = np.random.rand(3, 3, 3, 2).astype(np.float32)
    output_shape12 = np.array([1, 3, 3, 2]).astype(np.int32)
    strides12 = 1
    padding12 = 'VALID'
    data_format12 = 'NHWC'
    dilations12 = 1
    name12 = "conv_transpose12"

    input_dict12 = {
        "input": input12,
        "filters": filters12,
        "output_shape": output_shape12,
        "strides": strides12,
        "padding": padding12,
        "data_format": data_format12,
        "dilations": dilations12,
        "name": name12
    }
    list_of_inputs.append(copy.deepcopy(input_dict12))
    
    # Input 13: Dilation > 1, adjusted output shape, matching depths
    input13 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filters13 = np.random.rand(3, 3, 3, 2).astype(np.float32)
    output_shape13 = np.array([1, 9, 9, 2]).astype(np.int32)
    strides13 = 1
    padding13 = 'SAME'
    data_format13 = 'NHWC'
    dilations13 = 2
    name13 = "conv_transpose13"

    input_dict13 = {
        "input": input13,
        "filters": filters13,
        "output_shape": output_shape13,
        "strides": strides13,
        "padding": padding13,
        "data_format": data_format13,
        "dilations": dilations13,
        "name": name13
    }
    list_of_inputs.append(copy.deepcopy(input_dict13))

    # Input 14: Stride > 1 and valid output shape, matching depths
    input14 = np.random.rand(1, 4, 4, 3).astype(np.float32)
    filters14 = np.random.rand(2, 2, 3, 2).astype(np.float32)
    output_shape14 = np.array([1, 7, 7, 2]).astype(np.int32)
    strides14 = 2
    padding14 = 'SAME'
    data_format14 = 'NHWC'
    dilations14 = 1
    name14 = "conv_transpose14"

    input_dict14 = {
        "input": input14,
        "filters": filters14,
        "output_shape": output_shape14,
        "strides": strides14,
        "padding": padding14,
        "data_format": data_format14,
        "dilations": dilations14,
        "name": name14
    }
    list_of_inputs.append(copy.deepcopy(input_dict14))
    
    # Input 15: VALID padding, Stride > 1 and valid output shape, matching depths
    input15 = np.random.rand(1, 4, 4, 3).astype(np.float32)
    filters15 = np.random.rand(2, 2, 3, 2).astype(np.float32)
    output_shape15 = np.array([1, 5, 5, 2]).astype(np.int32)
    strides15 = 2
    padding15 = 'VALID'
    data_format15 = 'NHWC'
    dilations15 = 1
    name15 = "conv_transpose15"

    input_dict15 = {
        "input": input15,
        "filters": filters15,
        "output_shape": output_shape15,
        "strides": strides15,
        "padding": padding15,
        "data_format": data_format15,
        "dilations": dilations15,
        "name": name15
    }
    list_of_inputs.append(copy.deepcopy(input_dict15))

    # Input 16: Stride 1, VALID, Dilation, small sizes, matching dimensions
    input16 = np.random.rand(1, 3, 3, 1).astype(np.float32)
    filters16 = np.random.rand(2, 2, 1, 1).astype(np.float32)
    output_shape16 = np.array([1, 4, 4, 1]).astype(np.int32)
    strides16 = 1
    padding16 = 'VALID'
    data_format16 = 'NHWC'
    dilations16 = 1
    name16 = "conv_transpose16"

    input_dict16 = {
        "input": input16,
        "filters": filters16,
        "output_shape": output_shape16,
        "strides": strides16,
        "padding": padding16,
        "data_format": data_format16,
        "dilations": dilations16,
        "name": name16
    }
    list_of_inputs.append(copy.deepcopy(input_dict16))

    # Input 17 : Stride 2, VALID, Dilation 1, small sizes, matching dimensions
    input17 = np.random.rand(1, 3, 3, 1).astype(np.float32)
    filters17 = np.random.rand(2, 2, 1, 1).astype(np.float32)
    output_shape17 = np.array([1, 5, 5, 1]).astype(np.int32)
    strides17 = 2
    padding17 = 'VALID'
    data_format17 = 'NHWC'
    dilations17 = 1
    name17 = "conv_transpose17"

    input_dict17 = {
        "input": input17,
        "filters": filters17,
        "output_shape": output_shape17,
        "strides": strides17,
        "padding": padding17,
        "data_format": data_format17,
        "dilations": dilations17,
        "name": name17
    }
    list_of_inputs.append(copy.deepcopy(input_dict17))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.conv_transpose_1"] = tf_nn_conv_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.conv_transpose_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv_transpose_1'.")

check_valid('tf.nn.conv_transpose', generated_inputs['tf.nn.conv_transpose_1'], lib="tf", suffix=1)
