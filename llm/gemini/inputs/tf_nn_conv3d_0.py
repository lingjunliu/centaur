
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv3d_inputs():
    list_of_inputs = []

    # Case 1: Simple valid case with NDHWC, same padding, float32
    input_val = np.random.randn(1, 2, 2, 2, 1).astype(np.float32)
    filters = np.random.randn(1, 1, 1, 1, 1).astype(np.float32)
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv1"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    # Case 2: Larger shapes, negative values, valid padding, float32
    input_val = np.random.randn(2, 3, 3, 3, 2).astype(np.float32)
    filters = np.random.randn(2, 2, 2, 2, 4).astype(np.float32)
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv2"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    # Case 3: Strides of 2 in spatial dimensions, same padding, float32
    input_val = np.random.randn(1, 4, 4, 4, 3).astype(np.float32)
    filters = np.random.randn(3, 3, 3, 3, 2).astype(np.float32)
    strides = [1, 2, 2, 2, 1]
    padding = "SAME"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv3"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    # Case 4: Multiple input/output channels, NDHWC, float32
    input_val = np.random.randn(1, 3, 3, 3, 2).astype(np.float32)
    filters = np.random.randn(2, 2, 2, 2, 3).astype(np.float32)
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv4"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    # Case 5: Float16 precision
    input_val = np.random.randn(1, 2, 2, 2, 1).astype(np.float16)
    filters = np.random.randn(1, 1, 1, 1, 1).astype(np.float16)
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv5"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    # Case 6: Float64 precision
    input_val = np.random.randn(1, 3, 3, 3, 1).astype(np.float64)
    filters = np.random.randn(2, 2, 2, 1, 2).astype(np.float64)
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv6"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    # Case 7: Dilation in spatial dimensions, NDHWC
    input_val = np.random.randn(1, 5, 5, 5, 1).astype(np.float32)
    filters = np.random.randn(2, 2, 2, 1, 1).astype(np.float32)
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    dilations = [1, 2, 2, 2, 1]
    name = "conv7"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    # Case 8: Heterogeneous strides (different stride for height/width)
    input_val = np.random.randn(2, 4, 4, 4, 3).astype(np.float32)
    filters = np.random.randn(1, 2, 3, 3, 2).astype(np.float32)
    strides = [1, 1, 2, 2, 1]
    padding = "VALID"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv8"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    # Case 9: Stride on depth dimension only
    input_val = np.random.randn(1, 6, 6, 6, 2).astype(np.float32)
    filters = np.random.randn(3, 3, 3, 2, 2).astype(np.float32)
    strides = [1, 2, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv9"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    # Case 10: Mixed positive and negative float32 values with simple strides
    input_val = np.random.uniform(-1, 1, (1, 3, 3, 3, 2)).astype(np.float32)
    filters = np.random.uniform(-1, 1, (2, 2, 2, 2, 1)).astype(np.float32)
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv10"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    return list_of_inputs

generated_inputs["tf.nn.conv3d"] = tf_nn_conv3d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.conv3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv3d'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.conv3d', generated_inputs['tf.nn.conv3d'], lib="tf", suffix=0)
