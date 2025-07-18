
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import math

def get_tf_raw_ops_max_pool_3d_grad_grad_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.MaxPool3DGradGrad function.
    """
    
    def _get_pad_val(dtype):
        if np.issubdtype(dtype, np.floating):
            return -np.inf
        elif np.issubdtype(dtype, np.integer):
            # This branch is not used by the generated inputs but kept for completeness
            return np.iinfo(dtype).min
        return 0

    def _numpy_max_pool_3d(orig_input, ksize, strides, padding, data_format):
        """
        A numpy implementation of 3D max pooling to generate orig_output.
        """
        if data_format == 'NCDHW':
            # This should not happen as CPU only supports NDHWC, but keeping logic
            input_ndhwc = np.transpose(orig_input, (0, 2, 3, 4, 1))
            k_d, k_h, k_w = ksize[2], ksize[3], ksize[4]
            s_d, s_h, s_w = strides[2], strides[3], strides[4]
        else:  # NDHWC
            input_ndhwc = orig_input
            k_d, k_h, k_w = ksize[1], ksize[2], ksize[3]
            s_d, s_h, s_w = strides[1], strides[2], strides[3]

        N, D_in, H_in, W_in, C = input_ndhwc.shape
        padded_input = input_ndhwc

        if padding == 'VALID':
            D_out = math.ceil((D_in - k_d + 1) / s_d)
            H_out = math.ceil((H_in - k_h + 1) / s_h)
            W_out = math.ceil((W_in - k_w + 1) / s_w)
        elif padding == 'SAME':
            D_out = math.ceil(D_in / s_d)
            H_out = math.ceil(H_in / s_h)
            W_out = math.ceil(W_in / s_w)
            
            pad_d_total = max(0, (D_out - 1) * s_d + k_d - D_in)
            pad_h_total = max(0, (H_out - 1) * s_h + k_h - H_in)
            pad_w_total = max(0, (W_out - 1) * s_w + k_w - W_in)

            pad_d_before = pad_d_total // 2
            pad_d_after = pad_d_total - pad_d_before
            pad_h_before = pad_h_total // 2
            pad_h_after = pad_h_total - pad_h_before
            pad_w_before = pad_w_total // 2
            pad_w_after = pad_w_total - pad_w_before

            pad_val = _get_pad_val(orig_input.dtype)
            padded_input = np.pad(input_ndhwc,
                                 ((0, 0), (pad_d_before, pad_d_after), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after), (0, 0)),
                                 mode='constant', constant_values=pad_val)
        else:
            raise ValueError("Padding must be 'VALID' or 'SAME'")

        output_shape_ndhwc = (N, D_out, H_out, W_out, C)
        output_ndhwc = np.zeros(output_shape_ndhwc, dtype=orig_input.dtype)

        for n in range(N):
            for c in range(C):
                for d in range(D_out):
                    for h in range(H_out):
                        for w in range(W_out):
                            d_start = d * s_d
                            h_start = h * s_h
                            w_start = w * s_w
                            
                            window = padded_input[n, d_start:d_start+k_d, h_start:h_start+k_h, w_start:w_start+k_w, c]
                            if window.size > 0:
                                output_ndhwc[n, d, h, w, c] = np.max(window)

        if data_format == 'NCDHW':
            return np.transpose(output_ndhwc, (0, 4, 1, 2, 3))
        else:
            return output_ndhwc

    list_of_inputs = []
    
    # Input 1: Basic case, NDHWC, VALID padding, float32
    orig_input_1 = np.arange(27, dtype=np.float32).reshape((1, 3, 3, 3, 1))
    ksize_1 = [1, 2, 2, 2, 1]
    strides_1 = [1, 1, 1, 1, 1]
    padding_1 = "VALID"
    data_format_1 = "NDHWC"
    orig_output_1 = _numpy_max_pool_3d(orig_input_1, ksize_1, strides_1, padding_1, data_format_1)
    grad_1 = np.random.rand(*orig_input_1.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_1, 'orig_output': orig_output_1, 'grad': grad_1,
        'ksize': ksize_1, 'strides': strides_1, 'padding': padding_1, 
        'data_format': data_format_1, 'name': 'test_1'
    }))

    # Input 2: NDHWC, SAME padding
    orig_input_2 = np.arange(27, dtype=np.float32).reshape((1, 3, 3, 3, 1))
    ksize_2 = [1, 2, 2, 2, 1]
    strides_2 = [1, 1, 1, 1, 1]
    padding_2 = "SAME"
    data_format_2 = "NDHWC"
    orig_output_2 = _numpy_max_pool_3d(orig_input_2, ksize_2, strides_2, padding_2, data_format_2)
    grad_2 = np.random.rand(*orig_input_2.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_2, 'orig_output': orig_output_2, 'grad': grad_2,
        'ksize': ksize_2, 'strides': strides_2, 'padding': padding_2, 
        'data_format': data_format_2, 'name': 'test_2'
    }))

    # Input 3: NDHWC format, VALID padding
    orig_input_3 = np.arange(1 * 4 * 4 * 4 * 2, dtype=np.float32).reshape((1, 4, 4, 4, 2))
    ksize_3 = [1, 2, 2, 2, 1]
    strides_3 = [1, 1, 1, 1, 1]
    padding_3 = "VALID"
    data_format_3 = "NDHWC"
    orig_output_3 = _numpy_max_pool_3d(orig_input_3, ksize_3, strides_3, padding_3, data_format_3)
    grad_3 = np.random.rand(*orig_input_3.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_3, 'orig_output': orig_output_3, 'grad': grad_3,
        'ksize': ksize_3, 'strides': strides_3, 'padding': padding_3, 
        'data_format': data_format_3, 'name': 'test_3'
    }))

    # Input 4: NDHWC format, SAME padding, strides > 1
    orig_input_4 = np.arange(1 * 5 * 5 * 5 * 1, dtype=np.float32).reshape((1, 5, 5, 5, 1))
    ksize_4 = [1, 3, 3, 3, 1]
    strides_4 = [1, 2, 2, 2, 1]
    padding_4 = "SAME"
    data_format_4 = "NDHWC"
    orig_output_4 = _numpy_max_pool_3d(orig_input_4, ksize_4, strides_4, padding_4, data_format_4)
    grad_4 = np.random.rand(*orig_input_4.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_4, 'orig_output': orig_output_4, 'grad': grad_4,
        'ksize': ksize_4, 'strides': strides_4, 'padding': padding_4, 
        'data_format': data_format_4, 'name': 'test_4'
    }))

    # Input 5: float32 dtype (previously int32)
    orig_input_5 = np.arange(27, dtype=np.float32).reshape((1, 3, 3, 3, 1))
    ksize_5 = [1, 3, 3, 3, 1]
    strides_5 = [1, 1, 1, 1, 1]
    padding_5 = "VALID"
    data_format_5 = "NDHWC"
    orig_output_5 = _numpy_max_pool_3d(orig_input_5, ksize_5, strides_5, padding_5, data_format_5)
    grad_5 = (np.random.rand(*orig_input_5.shape) * 10).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_5, 'orig_output': orig_output_5, 'grad': grad_5,
        'ksize': ksize_5, 'strides': strides_5, 'padding': padding_5, 
        'data_format': data_format_5, 'name': 'test_5'
    }))

    # Input 6: Larger batch and channels
    orig_input_6 = np.arange(2 * 3 * 3 * 3 * 2, dtype=np.float32).reshape((2, 3, 3, 3, 2))
    ksize_6 = [1, 2, 2, 2, 1]
    strides_6 = [1, 1, 1, 1, 1]
    padding_6 = "SAME"
    data_format_6 = "NDHWC"
    orig_output_6 = _numpy_max_pool_3d(orig_input_6, ksize_6, strides_6, padding_6, data_format_6)
    grad_6 = np.random.rand(*orig_input_6.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_6, 'orig_output': orig_output_6, 'grad': grad_6,
        'ksize': ksize_6, 'strides': strides_6, 'padding': padding_6, 
        'data_format': data_format_6, 'name': 'test_6'
    }))

    # Input 7: Negative values, float32
    orig_input_7 = np.arange(-50, 4, dtype=np.float32).reshape((1, 3, 3, 6, 1))
    ksize_7 = [1, 2, 2, 2, 1]
    strides_7 = [1, 1, 1, 1, 1]
    padding_7 = "VALID"
    data_format_7 = "NDHWC"
    orig_output_7 = _numpy_max_pool_3d(orig_input_7, ksize_7, strides_7, padding_7, data_format_7)
    grad_7 = (np.random.rand(*orig_input_7.shape) - 0.5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_7, 'orig_output': orig_output_7, 'grad': grad_7,
        'ksize': ksize_7, 'strides': strides_7, 'padding': padding_7, 
        'data_format': data_format_7, 'name': 'test_7'
    }))

    # Input 8: half (float16) dtype, NDHWC
    orig_input_8 = np.arange(2 * 4 * 4 * 4 * 3, dtype=np.float16).reshape((2, 4, 4, 4, 3))
    ksize_8 = [1, 2, 2, 2, 1]
    strides_8 = [1, 2, 2, 2, 1]
    padding_8 = "SAME"
    data_format_8 = "NDHWC"
    orig_output_8 = _numpy_max_pool_3d(orig_input_8, ksize_8, strides_8, padding_8, data_format_8)
    grad_8 = np.random.rand(*orig_input_8.shape).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_8, 'orig_output': orig_output_8, 'grad': grad_8,
        'ksize': ksize_8, 'strides': strides_8, 'padding': padding_8, 
        'data_format': data_format_8, 'name': 'test_8'
    }))

    # Input 9: Non-cubic shapes and kernels
    orig_input_9 = np.arange(1 * 5 * 6 * 7 * 3, dtype=np.float32).reshape((1, 5, 6, 7, 3))
    ksize_9 = [1, 3, 2, 4, 1]
    strides_9 = [1, 1, 2, 3, 1]
    padding_9 = "SAME"
    data_format_9 = "NDHWC"
    orig_output_9 = _numpy_max_pool_3d(orig_input_9, ksize_9, strides_9, padding_9, data_format_9)
    grad_9 = np.random.rand(*orig_input_9.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_9, 'orig_output': orig_output_9, 'grad': grad_9,
        'ksize': ksize_9, 'strides': strides_9, 'padding': padding_9, 
        'data_format': data_format_9, 'name': 'test_9'
    }))

    # Input 10: float32 dtype (previously int8)
    orig_input_10 = np.arange(-60, 65, dtype=np.float32).reshape((1, 5, 5, 5, 1))
    ksize_10 = [1, 2, 3, 2, 1]
    strides_10 = [1, 2, 1, 2, 1]
    padding_10 = "VALID"
    data_format_10 = "NDHWC"
    orig_output_10 = _numpy_max_pool_3d(orig_input_10, ksize_10, strides_10, padding_10, data_format_10)
    grad_10 = (np.random.rand(*orig_input_10.shape) * 20 - 10).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        'orig_input': orig_input_10, 'orig_output': orig_output_10, 'grad': grad_10,
        'ksize': ksize_10, 'strides': strides_10, 'padding': padding_10, 
        'data_format': data_format_10, 'name': 'test_10'
    }))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPool3DGradGrad"] = get_tf_raw_ops_max_pool_3d_grad_grad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MaxPool3DGradGrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPool3DGradGrad'.")

check_valid('tf.raw_ops.MaxPool3DGradGrad', generated_inputs['tf.raw_ops.MaxPool3DGradGrad'], lib="tf", suffix=0)
