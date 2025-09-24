
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_max_pool_grad_grad_with_argmax_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.MaxPoolGradGradWithArgmax.
    """
    list_of_inputs = []

    def get_output_size(in_size, k_size, stride, padding):
        if padding.upper() == 'VALID':
            return (in_size - k_size) // stride + 1
        elif padding.upper() == 'SAME':
            return (in_size + stride - 1) // stride
        raise ValueError("Invalid padding type")

    # Configuration templates based on supported kernels (float types for input/grad, int64 for argmax)
    configs = [
        # 1. Basic float32, VALID padding
        {'input_shape': [2, 10, 10, 3], 'ksize': [1, 2, 2, 1], 'strides': [1, 2, 2, 1], 'padding': 'VALID', 'include_batch_in_index': False, 'dtype': np.float32, 'argmax_dtype': np.int64},
        # 2. SAME padding
        {'input_shape': [2, 10, 10, 3], 'ksize': [1, 2, 2, 1], 'strides': [1, 2, 2, 1], 'padding': 'SAME', 'include_batch_in_index': False, 'dtype': np.float32, 'argmax_dtype': np.int64},
        # 3. include_batch_in_index=True
        {'input_shape': [2, 10, 10, 3], 'ksize': [1, 2, 2, 1], 'strides': [1, 2, 2, 1], 'padding': 'VALID', 'include_batch_in_index': True, 'dtype': np.float32, 'argmax_dtype': np.int64},
        # 4. float64 dtype
        {'input_shape': [1, 8, 8, 2], 'ksize': [1, 3, 3, 1], 'strides': [1, 1, 1, 1], 'padding': 'VALID', 'include_batch_in_index': False, 'dtype': np.float64, 'argmax_dtype': np.int64},
        # 5. Non-square ksize/strides
        {'input_shape': [1, 9, 5, 1], 'ksize': [1, 3, 2, 1], 'strides': [1, 2, 1, 1], 'padding': 'SAME', 'include_batch_in_index': False, 'dtype': np.float32, 'argmax_dtype': np.int64},
        # 6. Strides larger than ksize
        {'input_shape': [1, 8, 8, 1], 'ksize': [1, 2, 2, 1], 'strides': [1, 3, 3, 1], 'padding': 'SAME', 'include_batch_in_index': False, 'dtype': np.float32, 'argmax_dtype': np.int64},
        # 7. half (float16) dtype
        {'input_shape': [2, 5, 5, 5], 'ksize': [1, 3, 3, 1], 'strides': [1, 1, 1, 1], 'padding': 'VALID', 'include_batch_in_index': False, 'dtype': np.float16, 'argmax_dtype': np.int64},
        # 8. Batch size of 1
        {'input_shape': [1, 20, 20, 3], 'ksize': [1, 5, 5, 1], 'strides': [1, 5, 5, 1], 'padding': 'SAME', 'include_batch_in_index': False, 'dtype': np.float32, 'argmax_dtype': np.int64},
        # 9. Another float32 case with different shapes
        {'input_shape': [2, 12, 12, 1], 'ksize': [1, 4, 4, 1], 'strides': [1, 1, 1, 1], 'padding': 'VALID', 'include_batch_in_index': False, 'dtype': np.float32, 'argmax_dtype': np.int64},
        # 10. float64, SAME padding, include_batch_in_index=True
        {'input_shape': [3, 9, 9, 2], 'ksize': [1, 3, 3, 1], 'strides': [1, 3, 3, 1], 'padding': 'SAME', 'include_batch_in_index': True, 'dtype': np.float64, 'argmax_dtype': np.int64},
    ]

    for i, config in enumerate(configs):
        input_shape = config['input_shape']
        ksize = config['ksize']
        strides = config['strides']
        padding = config['padding']
        dtype = config['dtype']
        
        # Per documentation, 'grad' has the same shape as 'input'
        input_tensor = np.random.randn(*input_shape).astype(dtype)
        grad_tensor = np.random.randn(*input_shape).astype(dtype)

        # Calculate output shape for argmax
        out_h = get_output_size(input_shape[1], ksize[1], strides[1], padding)
        out_w = get_output_size(input_shape[2], ksize[2], strides[2], padding)
        output_shape = [input_shape[0], out_h, out_w, input_shape[3]]
        
        # Generate argmax tensor. Indices are into the flattened input tensor.
        if config['include_batch_in_index']:
            max_idx = np.prod(input_shape)
        else:
            max_idx = np.prod(input_shape[1:])
            
        argmax_tensor = np.random.randint(0, max_idx, size=output_shape).astype(config['argmax_dtype'])

        input_dict = {
            'input': input_tensor,
            'grad': grad_tensor,
            'argmax': argmax_tensor,
            'ksize': ksize,
            'strides': strides,
            'padding': padding,
            'include_batch_in_index': config['include_batch_in_index'],
            'name': f'test_{i+1}'
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPoolGradGradWithArgmax"] = tf_raw_ops_max_pool_grad_grad_with_argmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MaxPoolGradGradWithArgmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPoolGradGradWithArgmax'.")

check_valid('tf.raw_ops.MaxPoolGradGradWithArgmax', generated_inputs['tf.raw_ops.MaxPoolGradGradWithArgmax'], lib="tf", suffix=0)
