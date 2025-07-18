
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_quantizedmaxpool_inputs():
    """
    This function generates a list of valid inputs for the tf.raw_ops.QuantizedMaxPool op.
    Inputs are in numpy format to pass the test harness validation. The test harness is expected
    to correctly convert the numpy array to a quantized tensor using the provided min/max ranges.
    """
    list_of_inputs = []

    # Input 1: quint8 with VALID padding.
    input_dict_1 = {
        'input': np.array([[[[0],[1]],[[2],[3]]]], dtype=np.uint8),
        'min_input': np.array(0.0, dtype=np.float32),
        'max_input': np.array(10.0, dtype=np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'name': 'quint8_valid_padding'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: qint8 with SAME padding.
    input_dict_2 = {
        'input': np.arange(-8, 8, dtype=np.int8).reshape(1, 4, 4, 1),
        'min_input': np.array(-10.0, dtype=np.float32),
        'max_input': np.array(10.0, dtype=np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'name': 'qint8_same_padding'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: qint16 with larger batch size and depth.
    input_dict_3 = {
        'input': np.arange(-75, 75, dtype=np.int16).reshape(2, 5, 5, 3),
        'min_input': np.array(-100.0, dtype=np.float32),
        'max_input': np.array(100.0, dtype=np.float32),
        'ksize': [1, 3, 3, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'name': 'qint16_larger_batch'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: quint16 with different ksize/strides.
    input_dict_4 = {
        'input': np.arange(64, dtype=np.uint16).reshape(1, 8, 8, 1),
        'min_input': np.array(0.0, dtype=np.float32),
        'max_input': np.array(100.0, dtype=np.float32),
        'ksize': [1, 3, 3, 1],
        'strides': [1, 3, 3, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: qint32 with non-square ksize.
    input_dict_5 = {
        'input': np.arange(-12, 12, dtype=np.int32).reshape(1, 6, 4, 1),
        'min_input': np.array(-20.0, dtype=np.float32),
        'max_input': np.array(20.0, dtype=np.float32),
        'ksize': [1, 2, 3, 1],
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'qint32_nonsquare_ksize'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: quint8 with stride > ksize.
    input_dict_6 = {
        'input': (np.arange(49, dtype=np.uint8)).reshape(1, 7, 7, 1),
        'min_input': np.array(0.0, dtype=np.float32),
        'max_input': np.array(50.0, dtype=np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 3, 3, 1],
        'padding': 'VALID',
        'name': 'quint8_stride_gt_ksize'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: qint8 with all-zeros input.
    input_dict_7 = {
        'input': np.zeros((1, 4, 4, 3), dtype=np.int8),
        'min_input': np.array(-1.0, dtype=np.float32),
        'max_input': np.array(1.0, dtype=np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'name': 'qint8_all_zeros_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Global max pooling simulation (ksize = input size).
    input_dict_8 = {
        'input': np.arange(25, dtype=np.uint8).reshape(1, 5, 5, 1),
        'min_input': np.array(0.0, dtype=np.float32),
        'max_input': np.array(30.0, dtype=np.float32),
        'ksize': [1, 5, 5, 1],
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'quint8_global_max_pool'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: quint16 with non-unit depth stride
    input_dict_9 = {
        'input': np.arange(3 * 6 * 6 * 4, dtype=np.uint16).reshape(3, 6, 6, 4),
        'min_input': np.array(0.0, dtype=np.float32),
        'max_input': np.array(1000.0, dtype=np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 2],
        'padding': 'VALID',
        'name': 'quint16_non_unit_depth_stride'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: qint32 with large values and SAME padding
    input_dict_10 = {
        'input': np.array(np.linspace(-1e5, 1e5, 25), dtype=np.int32).reshape(1, 5, 5, 1),
        'min_input': np.array(-1.1e5, dtype=np.float32),
        'max_input': np.array(1.1e5, dtype=np.float32),
        'ksize': [1, 3, 3, 1],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'name': 'qint32_large_values_same'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedMaxPool"] = tf_raw_ops_quantizedmaxpool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedMaxPool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedMaxPool'.")

check_valid('tf.raw_ops.QuantizedMaxPool', generated_inputs['tf.raw_ops.QuantizedMaxPool'], lib="tf", suffix=0)
