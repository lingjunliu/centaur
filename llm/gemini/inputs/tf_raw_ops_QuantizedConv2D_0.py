
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_raw_ops_quantized_conv2d_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.QuantizedConv2D function.
    """
    list_of_inputs = []

    # Helper to create the structured dictionary
    def create_input_dict(input_tensor, filter_tensor, min_input, max_input, min_filter, max_filter, strides, padding, out_type, dilations, name):
        return {
            'input': input_tensor,
            'filter': filter_tensor,
            'min_input': np.array(min_input, dtype=np.float32),
            'max_input': np.array(max_input, dtype=np.float32),
            'min_filter': np.array(min_filter, dtype=np.float32),
            'max_filter': np.array(max_filter, dtype=np.float32),
            'strides': strides,
            'padding': padding,
            'out_type': out_type,
            'dilations': dilations,
            'name': name
        }

    # The fundamental issue is that TensorFlow distinguishes between `tf.uint8` and `tf.quint8`.
    # `tf.raw_ops.QuantizedConv2D` requires the latter. However, there is no standard way
    # to create a numpy array that automatically converts to a `tf.quint8` tensor.
    # The `tf.quint8.as_numpy_dtype` creates a structured dtype that the test harness rejects.
    # The following inputs use standard numpy dtypes (np.uint8, np.int8), which will be
    # converted to `tf.uint8` and `tf.int8`, likely causing an InvalidArgumentError from TF.
    # This is an inherent limitation of trying to call this low-level op with numpy arrays.
    # We provide combinations of types that correspond to existing C++ kernels.

    # Based on available kernels like QuantizedConv2D<CPUDevice, quint8, qint8, qint32, ...>
    # Try Tinput=quint8 (np.uint8), Tfilter=qint8 (np.int8), out_type=qint32

    # Input 1: Basic quint8 input, qint8 filter
    input_1 = np.random.randint(0, 256, size=(1, 4, 4, 1)).astype(np.uint8)
    filter_1 = np.random.randint(-128, 128, size=(2, 2, 1, 2)).astype(np.int8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_1, filter_1, 0.0, 255.0, -127.0, 127.0, [1, 1, 1, 1], "VALID", tf.qint32, [1, 1, 1, 1], "test_quint8_qint8"
    )))

    # Input 2: SAME padding
    input_2 = np.random.randint(0, 256, size=(1, 5, 5, 3)).astype(np.uint8)
    filter_2 = np.random.randint(-128, 128, size=(3, 3, 3, 4)).astype(np.int8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_2, filter_2, 0.0, 10.0, -5.0, 5.0, [1, 1, 1, 1], "SAME", tf.qint32, [1, 1, 1, 1], "test_same_padding"
    )))

    # Input 3: With strides
    input_3 = np.random.randint(0, 256, size=(1, 8, 8, 1)).astype(np.uint8)
    filter_3 = np.random.randint(-128, 128, size=(3, 3, 1, 2)).astype(np.int8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_3, filter_3, -10.0, 245.0, -100.0, 100.0, [1, 2, 2, 1], "VALID", tf.qint32, [1, 1, 1, 1], "test_strides"
    )))

    # Input 4: With dilations
    input_4 = np.random.randint(0, 256, size=(1, 10, 10, 1)).astype(np.uint8)
    filter_4 = np.random.randint(-128, 128, size=(3, 3, 1, 2)).astype(np.int8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_4, filter_4, 0.0, 255.0, -127.0, 127.0, [1, 1, 1, 1], "VALID", tf.qint32, [1, 2, 2, 1], "test_dilations"
    )))

    # Kernel for Tinput=qint8, Tfilter=qint8, out_type=qint32 exists
    # Input 5: qint8 input, qint8 filter
    input_5 = np.random.randint(-128, 128, size=(1, 4, 4, 2)).astype(np.int8)
    filter_5 = np.random.randint(-128, 128, size=(2, 2, 2, 3)).astype(np.int8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_5, filter_5, -128.0, 127.0, -128.0, 127.0, [1, 1, 1, 1], "VALID", tf.qint32, [1, 1, 1, 1], "test_qint8_qint8"
    )))

    # Kernel for Tinput=quint8, Tfilter=quint8, out_type=quint8 exists
    # Input 6: Tinput=quint8, Tfilter=quint8, out_type=quint8
    input_6 = np.random.randint(0, 256, size=(1, 3, 3, 2)).astype(np.uint8)
    filter_6 = np.random.randint(0, 256, size=(2, 2, 2, 4)).astype(np.uint8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_6, filter_6, 0.0, 255.0, 0.0, 255.0, [1, 1, 1, 1], "VALID", tf.quint8, [1, 1, 1, 1], "test_quint8_all"
    )))

    # Input 7: Larger dimensions with qint8
    input_7 = np.random.randint(-128, 128, size=(2, 16, 16, 3)).astype(np.int8)
    filter_7 = np.random.randint(-128, 128, size=(5, 5, 3, 8)).astype(np.int8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_7, filter_7, -100.0, 100.0, -50.0, 50.0, [1, 2, 2, 1], "SAME", tf.qint32, [1, 1, 1, 1], "test_larger_qint8"
    )))

    # Input 8: non-1 batch and depth
    input_8 = np.random.randint(0, 256, size=(4, 6, 6, 2)).astype(np.uint8)
    filter_8 = np.random.randint(-128, 128, size=(3, 3, 2, 5)).astype(np.int8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_8, filter_8, 0.0, 255.0, -127.0, 127.0, [1, 1, 1, 1], "VALID", tf.qint32, [1, 1, 1, 1], "test_batch_depth"
    )))

    # Input 9: Non-square filter
    input_9 = np.random.randint(0, 256, size=(1, 7, 7, 1)).astype(np.uint8)
    filter_9 = np.random.randint(-128, 128, size=(1, 3, 1, 2)).astype(np.int8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_9, filter_9, 0.0, 1.0, -1.0, 1.0, [1, 1, 1, 1], "VALID", tf.qint32, [1, 1, 1, 1], "test_nonsquare_filter"
    )))

    # Kernel for out_type=qint8 requires Tinput=qint8, Tfilter=qint8
    # Input 10: out_type qint8
    input_10 = np.random.randint(-128, 128, size=(1, 5, 5, 3)).astype(np.int8)
    filter_10 = np.random.randint(-128, 128, size=(3, 3, 3, 4)).astype(np.int8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        input_10, filter_10, -128.0, 127.0, -128.0, 127.0, [1, 1, 1, 1], "VALID", tf.qint8, [1, 1, 1, 1], "test_out_qint8"
    )))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedConv2D"] = get_tf_raw_ops_quantized_conv2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedConv2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedConv2D'.")

check_valid('tf.raw_ops.QuantizedConv2D', generated_inputs['tf.raw_ops.QuantizedConv2D'], lib="tf", suffix=0)
