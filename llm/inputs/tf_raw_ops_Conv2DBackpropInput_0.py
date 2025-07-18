
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gen_tf_raw_ops_Conv2DBackpropInput_inputs():
    list_of_inputs = []

    # Helper function to create an input dictionary, always using NHWC and no dilations
    # as they are not supported on CPU.
    def create_input_dict(input_sizes, filter_shape, out_backprop_shape, dtype, strides, padding, use_cudnn_on_gpu=True, explicit_paddings=[], name='test'):
        return {
            'input_sizes': np.array(input_sizes, dtype=np.int32),
            'filter': np.ones(filter_shape, dtype=dtype),
            'out_backprop': np.ones(out_backprop_shape, dtype=dtype),
            'strides': strides,
            'padding': padding,
            'use_cudnn_on_gpu': use_cudnn_on_gpu,
            'explicit_paddings': explicit_paddings,
            'data_format': 'NHWC',
            'dilations': [1, 1, 1, 1],
            'name': name
        }

    # Input 1: Basic case, NHWC, VALID padding
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_sizes=[1, 5, 5, 3],
        filter_shape=[3, 3, 3, 8],
        out_backprop_shape=[1, 3, 3, 8],
        dtype=np.float32,
        strides=[1, 1, 1, 1],
        padding='VALID'
    )))

    # Input 2: NHWC, SAME padding, stride=2
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_sizes=[2, 7, 7, 1],
        filter_shape=[3, 3, 1, 4],
        out_backprop_shape=[2, 4, 4, 4],
        dtype=np.float64,
        strides=[1, 2, 2, 1],
        padding='SAME'
    )))

    # Input 3: NHWC, SAME padding, different dimensions
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_sizes=[1, 6, 8, 3],
        filter_shape=[2, 2, 3, 5],
        out_backprop_shape=[1, 6, 8, 5],
        dtype=np.float64,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )))

    # Input 4: NHWC, VALID padding, stride=2, float16
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_sizes=[2, 7, 7, 1],
        filter_shape=[3, 3, 1, 4],
        out_backprop_shape=[2, 3, 3, 4],
        dtype=np.float16,
        strides=[1, 2, 2, 1],
        padding='VALID'
    )))

    # Input 5: VALID padding, int32
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_sizes=[1, 10, 10, 2],
        filter_shape=[3, 3, 2, 5],
        out_backprop_shape=[1, 8, 8, 5],
        dtype=np.int32,
        strides=[1, 1, 1, 1],
        padding='VALID'
    )))

    # Input 6: SAME padding, simple case
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_sizes=[1, 9, 9, 1],
        filter_shape=[2, 2, 1, 4],
        out_backprop_shape=[1, 9, 9, 4],
        dtype=np.float32,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )))

    # Input 7: VALID padding, with strides
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_sizes=[1, 13, 13, 1],
        filter_shape=[3, 3, 1, 2],
        out_backprop_shape=[1, 6, 6, 2],
        dtype=np.float64,
        strides=[1, 2, 2, 1],
        padding='VALID'
    )))

    # Input 8: EXPLICIT padding
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_sizes=[1, 5, 5, 3],
        filter_shape=[3, 3, 3, 8],
        out_backprop_shape=[1, 5, 7, 8],
        dtype=np.float32,
        strides=[1, 1, 1, 1],
        padding='EXPLICIT',
        explicit_paddings=[0, 0, 1, 1, 2, 2, 0, 0]
    )))

    # Input 9: EXPLICIT padding, float16
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_sizes=[1, 6, 6, 2],
        filter_shape=[3, 3, 2, 4],
        out_backprop_shape=[1, 7, 7, 4],
        dtype=np.float16,
        strides=[1, 1, 1, 1],
        padding='EXPLICIT',
        explicit_paddings=[0, 0, 2, 1, 1, 2, 0, 0]
    )))

    # Input 10: VALID, complex strides
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_sizes=[3, 15, 12, 4],
        filter_shape=[4, 3, 4, 6],
        out_backprop_shape=[3, 4, 5, 6],
        dtype=np.float32,
        strides=[1, 3, 2, 1],
        padding='VALID'
    )))

    # Input 11: use_cudnn_on_gpu=False
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_sizes=[1, 5, 5, 3],
        filter_shape=[3, 3, 3, 8],
        out_backprop_shape=[1, 3, 3, 8],
        dtype=np.float32,
        strides=[1, 1, 1, 1],
        padding='VALID',
        use_cudnn_on_gpu=False
    )))
    
    # Input 12: Unequal height/width, SAME padding, strided
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_sizes=[1, 8, 6, 2],
        filter_shape=[3, 4, 2, 5],
        out_backprop_shape=[1, 4, 6, 5],
        dtype=np.float32,
        strides=[1, 2, 1, 1],
        padding='SAME'
    )))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Conv2DBackpropInput"] = gen_tf_raw_ops_Conv2DBackpropInput_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Conv2DBackpropInput' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Conv2DBackpropInput'.")

check_valid('tf.raw_ops.Conv2DBackpropInput', generated_inputs['tf.raw_ops.Conv2DBackpropInput'], lib="tf", suffix=0)
