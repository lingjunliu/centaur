
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_atrous_conv2d_transpose_inputs():
    list_of_inputs = []

    # Input 1: Basic Float32, Padding 'SAME', Rate 2
    value = np.random.randn(2, 5, 5, 3).astype(np.float32)
    filters = np.random.randn(3, 3, 4, 3).astype(np.float32)
    output_shape = np.array([2, 5, 5, 4], dtype=np.int32)
    rate = 2
    padding = 'SAME'
    name = 'atrous_conv_transpose_1'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 2: Float32, Padding 'VALID', Rate 1
    value = np.random.randn(1, 4, 4, 2).astype(np.float32)
    filters = np.random.randn(3, 3, 5, 2).astype(np.float32)
    output_shape = np.array([1, 6, 6, 5], dtype=np.int32)
    rate = 1
    padding = 'VALID'
    name = 'atrous_conv_transpose_2'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 3: Float32, Padding 'VALID', Rate 2
    value = np.random.randn(2, 3, 3, 1).astype(np.float32)
    filters = np.random.randn(2, 2, 2, 1).astype(np.float32)
    output_shape = np.array([2, 5, 5, 2], dtype=np.int32)
    rate = 2
    padding = 'VALID'
    name = 'atrous_conv_transpose_3'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 4: Large Rate, Padding 'SAME', Rate 3
    value = np.random.randn(1, 8, 8, 4).astype(np.float32)
    filters = np.random.randn(3, 3, 2, 4).astype(np.float32)
    output_shape = np.array([1, 8, 8, 2], dtype=np.int32)
    rate = 3
    padding = 'SAME'
    name = 'atrous_conv_transpose_4'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 5: Float64 type
    value = np.random.randn(3, 10, 10, 2).astype(np.float64)
    filters = np.random.randn(5, 5, 3, 2).astype(np.float64)
    output_shape = np.array([3, 10, 10, 3], dtype=np.int32)
    rate = 1
    padding = 'SAME'
    name = 'atrous_conv_transpose_5'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 6: Asymmetric Filters, Padding 'VALID', Rate 3
    value = np.random.randn(1, 2, 2, 1).astype(np.float32)
    filters = np.random.randn(2, 3, 1, 1).astype(np.float32)
    output_shape = np.array([1, 5, 8, 1], dtype=np.int32)
    rate = 3
    padding = 'VALID'
    name = 'atrous_conv_transpose_6'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 7: Negative values
    value = np.random.uniform(-10.0, -1.0, (1, 3, 3, 1)).astype(np.float32)
    filters = np.random.uniform(-5.0, -0.5, (2, 2, 1, 1)).astype(np.float32)
    output_shape = np.array([1, 3, 3, 1], dtype=np.int32)
    rate = 2
    padding = 'SAME'
    name = 'atrous_conv_transpose_7'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 8: 1x1 base resolution, high rate
    value = np.random.randn(1, 1, 1, 1).astype(np.float32)
    filters = np.random.randn(2, 2, 1, 1).astype(np.float32)
    output_shape = np.array([1, 6, 6, 1], dtype=np.int32)
    rate = 5
    padding = 'VALID'
    name = 'atrous_conv_transpose_8'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 9: Large batch and high channel sizes
    value = np.random.randn(16, 14, 14, 32).astype(np.float32)
    filters = np.random.randn(3, 3, 64, 32).astype(np.float32)
    output_shape = np.array([16, 14, 14, 64], dtype=np.int32)
    rate = 2
    padding = 'SAME'
    name = 'atrous_conv_transpose_9'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 10: 1x1 filter size, rate 2, Padding 'VALID'
    value = np.random.randn(2, 5, 5, 16).astype(np.float32)
    filters = np.random.randn(1, 1, 8, 16).astype(np.float32)
    output_shape = np.array([2, 5, 5, 8], dtype=np.int32)
    rate = 2
    padding = 'VALID'
    name = 'atrous_conv_transpose_10'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    return list_of_inputs

generated_inputs["tf.nn.atrous_conv2d_transpose"] = tf_nn_atrous_conv2d_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.atrous_conv2d_transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.atrous_conv2d_transpose'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.atrous_conv2d_transpose', generated_inputs['tf.nn.atrous_conv2d_transpose'], lib="tf", suffix=0)
