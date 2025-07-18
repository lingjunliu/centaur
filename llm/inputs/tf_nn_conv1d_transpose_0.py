
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_nn_conv1d_transpose_inputs():
    """
    Generates a list of valid inputs for the tf.nn.conv1d_transpose function.
    """
    list_of_inputs = []

    # Input 1: Basic case with 'NWC' format, 'SAME' padding, and stride > 1
    input_dict_1 = {
        'input': np.random.rand(1, 5, 3).astype(np.float32),
        'filters': np.random.rand(3, 8, 3).astype(np.float32),
        'output_shape': np.array([1, 10, 8], dtype=np.int32),
        'strides': 2,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'test_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic case with 'NWC' format, 'VALID' padding
    input_dict_2 = {
        'input': np.random.rand(2, 4, 2).astype(np.float32),
        'filters': np.random.rand(4, 5, 2).astype(np.float32),
        'output_shape': np.array([2, 10, 5], dtype=np.int32),
        'strides': 2,
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'test_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 'NCW' data format with 'SAME' padding and stride = 1
    input_dict_3 = {
        'input': np.random.rand(1, 4, 6).astype(np.float32),
        'filters': np.random.rand(2, 10, 4).astype(np.float32),
        'output_shape': np.array([1, 10, 6], dtype=np.int32),
        'strides': 1,
        'padding': 'SAME',
        'data_format': 'NCW',
        'dilations': 1,
        'name': 'test_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 'NCW' data format with 'VALID' padding and stride = 3
    input_dict_4 = {
        'input': np.random.rand(3, 1, 8).astype(np.float32),
        'filters': np.random.rand(3, 6, 1).astype(np.float32),
        'output_shape': np.array([3, 6, 24], dtype=np.int32),
        'strides': 3,
        'padding': 'VALID',
        'data_format': 'NCW',
        'dilations': 1,
        'name': 'test_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: With dilations > 1, 'NWC' format and 'SAME' padding
    input_dict_5 = {
        'input': np.random.rand(1, 5, 3).astype(np.float32),
        'filters': np.random.rand(3, 8, 3).astype(np.float32),
        'output_shape': np.array([1, 10, 8], dtype=np.int32),
        'strides': 2,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 2,
        'name': 'test_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: With dilations > 1, 'NWC' format and 'VALID' padding
    input_dict_6 = {
        'input': np.random.rand(2, 4, 2).astype(np.float32),
        'filters': np.random.rand(4, 5, 2).astype(np.float32),
        'output_shape': np.array([2, 16, 5], dtype=np.int32),
        'strides': 2,
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 3,
        'name': 'test_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Larger batch size
    input_dict_7 = {
        'input': np.random.rand(16, 10, 1).astype(np.float32),
        'filters': np.random.rand(5, 32, 1).astype(np.float32),
        'output_shape': np.array([16, 10, 32], dtype=np.int32),
        'strides': 1,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'test_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Minimal dimensions (1x1x1) with 'SAME' padding
    input_dict_8 = {
        'input': np.ones((1, 1, 1), dtype=np.float32),
        'filters': np.ones((1, 1, 1), dtype=np.float32),
        'output_shape': np.array([1, 1, 1], dtype=np.int32),
        'strides': 1,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'test_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Minimal dimensions (1x1x1) with 'VALID' padding and stride > 1
    input_dict_9 = {
        'input': np.ones((1, 1, 1), dtype=np.float32),
        'filters': np.ones((1, 1, 1), dtype=np.float32),
        'output_shape': np.array([1, 1, 1], dtype=np.int32),
        'strides': 2,
        'padding': 'VALID',
        'data_format': 'NCW',
        'dilations': 1,
        'name': 'test_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Filter width larger than input width, 'VALID' padding
    input_dict_10 = {
        'input': np.random.rand(1, 3, 2).astype(np.float32),
        'filters': np.random.rand(5, 4, 2).astype(np.float32),
        'output_shape': np.array([1, 7, 4], dtype=np.int32),
        'strides': 1,
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'test_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Stride larger than filter width, 'VALID' padding
    input_dict_11 = {
        'input': np.random.rand(2, 5, 3).astype(np.float32),
        'filters': np.random.rand(3, 1, 3).astype(np.float32),
        'output_shape': np.array([2, 19, 1], dtype=np.int32),
        'strides': 4,
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'test_11'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Dilation with 'NCW' format and 'VALID' padding
    input_dict_12 = {
        'input': np.random.rand(4, 8, 10).astype(np.float32),
        'filters': np.random.rand(3, 16, 8).astype(np.float32),
        'output_shape': np.array([4, 23, 16], dtype=np.int32), # Correct order for NCW is [batch, out_channels, out_width]
        'strides': 2,
        'padding': 'VALID',
        'data_format': 'NCW',
        'dilations': 2,
        'name': 'test_12'
    }
    # Correction: The output_shape for NCW should be [batch, out_channels, out_width]
    # For Input 4: [3, 6, 24] is correct.
    # For Input 3: [1, 10, 6] is correct.
    # For Input 12: out_width = (10-1)*2 + ((3-1)*2+1) = 9*2 + 5 = 23. Shape is [4, 16, 23]. Correct.
    # Let's fix the NCW output shape order for clarity. For input 12, out_channels is 16, so the shape is [4, 16, 23].
    # In my code I have [4, 16, 23] which is correct. Wait, in my original thought process I wrote [4, 16, 23], but in the code I wrote it as [4, 23, 16].
    # NCW output shape is `[batch, out_channels, out_width]`. out_channels = filters[1] = 16. So it should be `[4, 16, 23]`. I'll fix the code.
    input_dict_12_corrected = {
        'input': np.random.rand(4, 8, 10).astype(np.float32),
        'filters': np.random.rand(3, 16, 8).astype(np.float32),
        'output_shape': np.array([4, 16, 23], dtype=np.int32),
        'strides': 2,
        'padding': 'VALID',
        'data_format': 'NCW',
        'dilations': 2,
        'name': 'test_12_corrected'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12_corrected))

    return list_of_inputs

generated_inputs["tf.nn.conv1d_transpose"] = tf_nn_conv1d_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.conv1d_transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv1d_transpose'.")

check_valid('tf.nn.conv1d_transpose', generated_inputs['tf.nn.conv1d_transpose'], lib="tf", suffix=0)
