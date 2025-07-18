
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_max_pool_grad_grad_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.MaxPoolGradGrad function.
    """
    list_of_inputs = []

    def np_max_pool(input_tensor, ksize, strides, padding, data_format):
        """
        A numpy implementation of max pooling to generate valid orig_output.
        """
        if data_format == "NHWC":
            h_axis, w_axis, c_axis = 1, 2, 3
        else:
            raise ValueError("Only NHWC format is supported for CPU execution.")

        k_h, k_w = ksize[h_axis], ksize[w_axis]
        s_h, s_w = strides[h_axis], strides[w_axis]

        in_shape = input_tensor.shape
        in_h, in_w = in_shape[h_axis], in_shape[w_axis]

        if padding == "VALID":
            out_h = (in_h - k_h) // s_h + 1
            out_w = (in_w - k_w) // s_w + 1
            pad_h_before, pad_h_after = 0, 0
            pad_w_before, pad_w_after = 0, 0
        else:  # SAME
            out_h = (in_h + s_h - 1) // s_h
            out_w = (in_w + s_w - 1) // s_w
            pad_h_needed = max(0, (out_h - 1) * s_h + k_h - in_h)
            pad_w_needed = max(0, (out_w - 1) * s_w + k_w - in_w)
            pad_h_before = pad_h_needed // 2
            pad_h_after = pad_h_needed - pad_h_before
            pad_w_before = pad_w_needed // 2
            pad_w_after = pad_w_needed - pad_w_before

        pad_width = ((0, 0), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after), (0, 0))

        dtype = input_tensor.dtype
        if np.issubdtype(dtype, np.integer):
            pad_value = np.iinfo(dtype).min
        else:
            pad_value = -np.inf

        padded_input = np.pad(input_tensor, pad_width, mode='constant', constant_values=pad_value)

        out_shape = list(in_shape)
        out_shape[h_axis], out_shape[w_axis] = out_h, out_w
        output_tensor = np.zeros(out_shape, dtype=dtype)

        N, _, _, C = in_shape
        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start, w_start = i * s_h, j * s_w
                        window = padded_input[n, h_start:h_start+k_h, w_start:w_start+k_w, c]
                        output_tensor[n, i, j, c] = np.max(window)
        return output_tensor

    # Case 1: Basic NHWC, float32, VALID
    orig_input_1 = np.random.randn(1, 4, 4, 1).astype(np.float32)
    ksize_1 = [1, 2, 2, 1]
    strides_1 = [1, 2, 2, 1]
    padding_1 = "VALID"
    data_format_1 = "NHWC"
    orig_output_1 = np_max_pool(orig_input_1, ksize_1, strides_1, padding_1, data_format_1)
    grad_1 = np.random.randn(*orig_input_1.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_1, 'orig_output': orig_output_1, 'grad': grad_1,
        'ksize': ksize_1, 'strides': strides_1, 'padding': padding_1, 'data_format': data_format_1, 'name': "case_1"
    }))

    # Case 2: Basic NHWC, float32, SAME
    orig_input_2 = np.random.randn(1, 3, 5, 1).astype(np.float32)
    ksize_2 = [1, 2, 2, 1]
    strides_2 = [1, 1, 1, 1]
    padding_2 = "SAME"
    data_format_2 = "NHWC"
    orig_output_2 = np_max_pool(orig_input_2, ksize_2, strides_2, padding_2, data_format_2)
    grad_2 = np.random.randn(*orig_input_2.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_2, 'orig_output': orig_output_2, 'grad': grad_2,
        'ksize': ksize_2, 'strides': strides_2, 'padding': padding_2, 'data_format': data_format_2, 'name': "case_2"
    }))

    # Case 3: NHWC, float64, VALID (Fixed from NCHW)
    orig_input_3 = np.random.randn(1, 6, 6, 1).astype(np.float64)
    ksize_3 = [1, 3, 3, 1]
    strides_3 = [1, 3, 3, 1]
    padding_3 = "VALID"
    data_format_3 = "NHWC"
    orig_output_3 = np_max_pool(orig_input_3, ksize_3, strides_3, padding_3, data_format_3)
    grad_3 = np.random.randn(*orig_input_3.shape).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_3, 'orig_output': orig_output_3, 'grad': grad_3,
        'ksize': ksize_3, 'strides': strides_3, 'padding': padding_3, 'data_format': data_format_3, 'name': "case_3"
    }))

    # Case 4: NHWC, float16, SAME (Fixed from NCHW)
    orig_input_4 = np.random.randn(1, 7, 7, 3).astype(np.float16)
    ksize_4 = [1, 2, 2, 1]
    strides_4 = [1, 2, 2, 1]
    padding_4 = "SAME"
    data_format_4 = "NHWC"
    orig_output_4 = np_max_pool(orig_input_4, ksize_4, strides_4, padding_4, data_format_4)
    grad_4 = np.random.randn(*orig_input_4.shape).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_4, 'orig_output': orig_output_4, 'grad': grad_4,
        'ksize': ksize_4, 'strides': strides_4, 'padding': padding_4, 'data_format': data_format_4, 'name': "case_4"
    }))

    # Case 5: int32 with negative numbers, NHWC, SAME
    orig_input_5 = np.random.randint(-100, 100, size=(1, 5, 5, 2)).astype(np.int32)
    ksize_5 = [1, 3, 3, 1]
    strides_5 = [1, 1, 1, 1]
    padding_5 = "SAME"
    data_format_5 = "NHWC"
    orig_output_5 = np_max_pool(orig_input_5, ksize_5, strides_5, padding_5, data_format_5)
    grad_5 = np.random.randint(-10, 10, size=orig_input_5.shape).astype(np.int32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_5, 'orig_output': orig_output_5, 'grad': grad_5,
        'ksize': ksize_5, 'strides': strides_5, 'padding': padding_5, 'data_format': data_format_5, 'name': "case_5"
    }))

    # Case 6: uint8, NHWC, VALID (Fixed from NCHW)
    orig_input_6 = np.random.randint(0, 255, size=(2, 8, 8, 3), dtype=np.uint8)
    ksize_6 = [1, 4, 4, 1]
    strides_6 = [1, 4, 4, 1]
    padding_6 = "VALID"
    data_format_6 = "NHWC"
    orig_output_6 = np_max_pool(orig_input_6, ksize_6, strides_6, padding_6, data_format_6)
    grad_6 = np.random.randint(0, 10, size=orig_input_6.shape, dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_6, 'orig_output': orig_output_6, 'grad': grad_6,
        'ksize': ksize_6, 'strides': strides_6, 'padding': padding_6, 'data_format': data_format_6, 'name': "case_6"
    }))
    
    # Case 7: Non-square ksize/strides, NHWC, VALID
    orig_input_7 = np.random.randn(1, 7, 6, 1).astype(np.float32)
    ksize_7 = [1, 3, 2, 1]
    strides_7 = [1, 2, 1, 1]
    padding_7 = "VALID"
    data_format_7 = "NHWC"
    orig_output_7 = np_max_pool(orig_input_7, ksize_7, strides_7, padding_7, data_format_7)
    grad_7 = np.random.randn(*orig_input_7.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_7, 'orig_output': orig_output_7, 'grad': grad_7,
        'ksize': ksize_7, 'strides': strides_7, 'padding': padding_7, 'data_format': data_format_7, 'name': "case_7"
    }))

    # Case 8: int8, NHWC, SAME
    orig_input_8 = np.random.randint(-128, 127, size=(1, 4, 4, 3), dtype=np.int8)
    ksize_8 = [1, 2, 2, 1]
    strides_8 = [1, 1, 1, 1]
    padding_8 = "SAME"
    data_format_8 = "NHWC"
    orig_output_8 = np_max_pool(orig_input_8, ksize_8, strides_8, padding_8, data_format_8)
    grad_8 = np.random.randint(-5, 5, size=orig_input_8.shape, dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_8, 'orig_output': orig_output_8, 'grad': grad_8,
        'ksize': ksize_8, 'strides': strides_8, 'padding': padding_8, 'data_format': data_format_8, 'name': "case_8"
    }))

    # Case 9: int64, NHWC, VALID (Fixed from NCHW)
    orig_input_9 = np.random.randint(-1e9, 1e9, size=(1, 9, 9, 2), dtype=np.int64)
    ksize_9 = [1, 3, 3, 1]
    strides_9 = [1, 2, 2, 1]
    padding_9 = "VALID"
    data_format_9 = "NHWC"
    orig_output_9 = np_max_pool(orig_input_9, ksize_9, strides_9, padding_9, data_format_9)
    grad_9 = np.random.randint(-100, 100, size=orig_input_9.shape, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_9, 'orig_output': orig_output_9, 'grad': grad_9,
        'ksize': ksize_9, 'strides': strides_9, 'padding': padding_9, 'data_format': data_format_9, 'name': "case_9"
    }))

    # Case 10: Identity pooling ksize=[1,1,1,1], NHWC, SAME
    orig_input_10 = np.arange(16).reshape(1, 4, 4, 1).astype(np.float32)
    ksize_10 = [1, 1, 1, 1]
    strides_10 = [1, 1, 1, 1]
    padding_10 = "SAME"
    data_format_10 = "NHWC"
    orig_output_10 = np_max_pool(orig_input_10, ksize_10, strides_10, padding_10, data_format_10)
    grad_10 = np.random.randn(*orig_input_10.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_10, 'orig_output': orig_output_10, 'grad': grad_10,
        'ksize': ksize_10, 'strides': strides_10, 'padding': padding_10, 'data_format': data_format_10, 'name': "case_10"
    }))

    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPoolGradGrad"] = get_max_pool_grad_grad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MaxPoolGradGrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPoolGradGrad'.")

check_valid('tf.raw_ops.MaxPoolGradGrad', generated_inputs['tf.raw_ops.MaxPoolGradGrad'], lib="tf", suffix=0)
