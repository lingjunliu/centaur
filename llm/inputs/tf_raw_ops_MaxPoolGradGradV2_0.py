
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import math

def get_tf_raw_ops_max_pool_grad_grad_v2_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.MaxPoolGradGradV2 function.
    """
    list_of_inputs = []

    def numpy_max_pool(input_tensor, ksize_list, strides_list, padding, data_format):
        """A numpy implementation of max pooling to generate valid orig_output."""
        input_copy = input_tensor.copy()

        if data_format == 'NCHW':
            input_copy = np.transpose(input_copy, (0, 2, 3, 1)) # to NHWC
            k_h, k_w = ksize_list[2], ksize_list[3]
            s_h, s_w = strides_list[2], strides_list[3]
        else: # NHWC
            k_h, k_w = ksize_list[1], ksize_list[2]
            s_h, s_w = strides_list[1], strides_list[2]

        batch_size, in_height, in_width, num_channels = input_copy.shape

        if padding == 'VALID':
            out_height = math.ceil((in_height - k_h + 1) / s_h)
            out_width = math.ceil((in_width - k_w + 1) / s_w)
            pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        elif padding == 'SAME':
            out_height = math.ceil(in_height / s_h)
            out_width = math.ceil(in_width / s_w)
            pad_needed_h = max(0, (out_height - 1) * s_h + k_h - in_height)
            pad_needed_w = max(0, (out_width - 1) * s_w + k_w - in_width)
            pad_top = pad_needed_h // 2
            pad_bottom = pad_needed_h - pad_top
            pad_left = pad_needed_w // 2
            pad_right = pad_needed_w - pad_left
        else:
            raise ValueError("Padding must be 'VALID' or 'SAME'")
        
        if any([pad_top, pad_bottom, pad_left, pad_right]):
            min_val = np.iinfo(input_copy.dtype).min if np.issubdtype(input_copy.dtype, np.integer) else -np.inf
            input_copy = np.pad(input_copy, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=min_val)
        
        output_tensor = np.zeros((batch_size, out_height, out_width, num_channels), dtype=input_copy.dtype)

        for b in range(batch_size):
            for i in range(out_height):
                for j in range(out_width):
                    h_start, w_start = i * s_h, j * s_w
                    window = input_copy[b, h_start:h_start+k_h, w_start:w_start+k_w, :]
                    output_tensor[b, i, j, :] = np.max(window, axis=(0, 1))

        if data_format == 'NCHW':
            output_tensor = np.transpose(output_tensor, (0, 3, 1, 2)) # back to NCHW
        
        return output_tensor

    # === Input 1: Basic case, float32, NHWC, VALID ===
    orig_input = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "VALID"
    data_format = "NHWC"
    orig_output = numpy_max_pool(orig_input, ksize, strides, padding, data_format)
    grad = np.random.randn(*orig_input.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input, 'orig_output': orig_output, 'grad': grad,
        'ksize': np.array(ksize, dtype=np.int32), 'strides': np.array(strides, dtype=np.int32),
        'padding': padding, 'data_format': data_format, 'name': 'test_1'
    }))

    # === Input 2: float32, NHWC, SAME padding ===
    orig_input = np.arange(25, dtype=np.float32).reshape(1, 5, 5, 1)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "SAME"
    data_format = "NHWC"
    orig_output = numpy_max_pool(orig_input, ksize, strides, padding, data_format)
    grad = np.random.randn(*orig_input.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input, 'orig_output': orig_output, 'grad': grad,
        'ksize': np.array(ksize, dtype=np.int32), 'strides': np.array(strides, dtype=np.int32),
        'padding': padding, 'data_format': data_format, 'name': 'test_2'
    }))

    # === Input 3: float32, NHWC, multiple channels ===
    orig_input = np.arange(1 * 4 * 4 * 3, dtype=np.float32).reshape(1, 4, 4, 3)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "VALID"
    data_format = "NHWC"
    orig_output = numpy_max_pool(orig_input, ksize, strides, padding, data_format)
    grad = np.random.randn(*orig_input.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input, 'orig_output': orig_output, 'grad': grad,
        'ksize': np.array(ksize, dtype=np.int32), 'strides': np.array(strides, dtype=np.int32),
        'padding': padding, 'data_format': data_format, 'name': 'test_3'
    }))

    # === Input 4: float64, NHWC, VALID ===
    orig_input = np.arange(16, dtype=np.float64).reshape(1, 4, 4, 1)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "VALID"
    data_format = "NHWC"
    orig_output = numpy_max_pool(orig_input, ksize, strides, padding, data_format)
    grad = np.random.randn(*orig_input.shape).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input, 'orig_output': orig_output, 'grad': grad,
        'ksize': np.array(ksize, dtype=np.int32), 'strides': np.array(strides, dtype=np.int32),
        'padding': padding, 'data_format': data_format, 'name': 'test_4'
    }))

    # === Input 5: int32 with negative values, NHWC, VALID ===
    orig_input = (np.arange(16, dtype=np.int32) - 8).reshape(1, 4, 4, 1)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "VALID"
    data_format = "NHWC"
    orig_output = numpy_max_pool(orig_input, ksize, strides, padding, data_format)
    grad = (np.arange(16, dtype=np.int32) - 4).reshape(1, 4, 4, 1)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input, 'orig_output': orig_output, 'grad': grad,
        'ksize': np.array(ksize, dtype=np.int32), 'strides': np.array(strides, dtype=np.int32),
        'padding': padding, 'data_format': data_format, 'name': 'test_5'
    }))

    # === Input 6: Larger kernel, NHWC, SAME ===
    orig_input = np.arange(36, dtype=np.float32).reshape(1, 6, 6, 1)
    ksize = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
    padding = "SAME"
    data_format = "NHWC"
    orig_output = numpy_max_pool(orig_input, ksize, strides, padding, data_format)
    grad = np.random.randn(*orig_input.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input, 'orig_output': orig_output, 'grad': grad,
        'ksize': np.array(ksize, dtype=np.int32), 'strides': np.array(strides, dtype=np.int32),
        'padding': padding, 'data_format': data_format, 'name': 'test_6'
    }))

    # === Input 7: Overlapping strides, NHWC, VALID ===
    orig_input = np.arange(25, dtype=np.float32).reshape(1, 5, 5, 1)
    ksize = [1, 3, 3, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"
    data_format = "NHWC"
    orig_output = numpy_max_pool(orig_input, ksize, strides, padding, data_format)
    grad = np.random.randn(*orig_input.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input, 'orig_output': orig_output, 'grad': grad,
        'ksize': np.array(ksize, dtype=np.int32), 'strides': np.array(strides, dtype=np.int32),
        'padding': padding, 'data_format': data_format, 'name': 'test_7'
    }))

    # === Input 8: Non-square kernel/stride, NHWC, VALID ===
    orig_input = np.arange(24, dtype=np.float32).reshape(1, 4, 6, 1)
    ksize = [1, 2, 3, 1]
    strides = [1, 2, 3, 1]
    padding = "VALID"
    data_format = "NHWC"
    orig_output = numpy_max_pool(orig_input, ksize, strides, padding, data_format)
    grad = np.random.randn(*orig_input.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input, 'orig_output': orig_output, 'grad': grad,
        'ksize': np.array(ksize, dtype=np.int32), 'strides': np.array(strides, dtype=np.int32),
        'padding': padding, 'data_format': data_format, 'name': 'test_8'
    }))

    # === Input 9: Batch size > 1, NHWC, VALID ===
    orig_input = np.arange(32, dtype=np.float32).reshape(2, 4, 4, 1)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "VALID"
    data_format = "NHWC"
    orig_output = numpy_max_pool(orig_input, ksize, strides, padding, data_format)
    grad = np.random.randn(*orig_input.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input, 'orig_output': orig_output, 'grad': grad,
        'ksize': np.array(ksize, dtype=np.int32), 'strides': np.array(strides, dtype=np.int32),
        'padding': padding, 'data_format': data_format, 'name': 'test_9'
    }))

    # === Input 10: Multiple channels, NHWC, SAME ===
    orig_input = np.arange(1 * 5 * 5 * 3, dtype=np.float32).reshape(1, 5, 5, 3)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "SAME"
    data_format = "NHWC"
    orig_output = numpy_max_pool(orig_input, ksize, strides, padding, data_format)
    grad = np.random.randn(*orig_input.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input, 'orig_output': orig_output, 'grad': grad,
        'ksize': np.array(ksize, dtype=np.int32), 'strides': np.array(strides, dtype=np.int32),
        'padding': padding, 'data_format': data_format, 'name': 'test_10'
    }))
    
    # === Input 11: half (float16) dtype, NHWC, SAME ===
    orig_input = np.arange(25, dtype=np.float16).reshape(1, 5, 5, 1)
    ksize = [1, 3, 3, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"
    data_format = "NHWC"
    orig_output = numpy_max_pool(orig_input, ksize, strides, padding, data_format)
    grad = np.random.randn(*orig_input.shape).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input, 'orig_output': orig_output, 'grad': grad,
        'ksize': np.array(ksize, dtype=np.int32), 'strides': np.array(strides, dtype=np.int32),
        'padding': padding, 'data_format': data_format, 'name': 'test_11'
    }))

    # === Input 12: uint8 dtype, NHWC, VALID ===
    orig_input = np.arange(1 * 4 * 4 * 3, dtype=np.uint8).reshape(1, 4, 4, 3)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "VALID"
    data_format = "NHWC"
    orig_output = numpy_max_pool(orig_input, ksize, strides, padding, data_format)
    grad = np.arange(1 * 4 * 4 * 3, dtype=np.uint8).reshape(1, 4, 4, 3)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input, 'orig_output': orig_output, 'grad': grad,
        'ksize': np.array(ksize, dtype=np.int32), 'strides': np.array(strides, dtype=np.int32),
        'padding': padding, 'data_format': data_format, 'name': 'test_12'
    }))

    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPoolGradGradV2"] = get_tf_raw_ops_max_pool_grad_grad_v2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MaxPoolGradGradV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPoolGradGradV2'.")

check_valid('tf.raw_ops.MaxPoolGradGradV2', generated_inputs['tf.raw_ops.MaxPoolGradGradV2'], lib="tf", suffix=0)
