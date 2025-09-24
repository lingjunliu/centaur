
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_quantized_max_pool_inputs():
    """
    This function generates a list of valid inputs for the
    tf.raw_ops.QuantizedMaxPool operation.
    Inputs are provided as NumPy arrays with standard integer dtypes to be
    compatible with the testing framework, which does not recognize
    quantized tf.DType objects like tf.qint8.
    """
    list_of_inputs = []

    # Input 1: Corresponds to quint8, VALID padding
    input_dict_1 = {
        'input': np.array([[[[10], [20]], [[30], [40]]]], dtype=np.uint8),
        'min_input': np.array(0.0, dtype=np.float32),
        'max_input': np.array(255.0, dtype=np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'name': 'test_quint8_valid'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Corresponds to qint8, SAME padding
    input_dict_2 = {
        'input': np.array([[[[-10], [20]], [[-30], [40]]]], dtype=np.int8),
        'min_input': np.array(-128.0, dtype=np.float32),
        'max_input': np.array(127.0, dtype=np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'name': 'test_qint8_same'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Corresponds to qint16, batch > 1
    input_dict_3 = {
        'input': np.arange(32, dtype=np.int16).reshape(2, 4, 4, 1),
        'min_input': np.array(-1000.0, dtype=np.float32),
        'max_input': np.array(1000.0, dtype=np.float32),
        'ksize': [1, 3, 3, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'name': 'test_qint16_batch'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Corresponds to quint16, non-square kernel
    input_dict_4 = {
        'input': np.arange(120, dtype=np.uint16).reshape(1, 6, 10, 2),
        'min_input': np.array(0.0, dtype=np.float32),
        'max_input': np.array(5000.0, dtype=np.float32),
        'ksize': [1, 2, 4, 1],
        'strides': [1, 1, 2, 1],
        'padding': 'SAME',
        'name': 'test_quint16_nonsquare'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Corresponds to qint32, large strides
    input_dict_5 = {
        'input': (np.arange(49, dtype=np.int32) - 24).reshape(1, 7, 7, 1),
        'min_input': np.array(-200000.0, dtype=np.float32),
        'max_input': np.array(200000.0, dtype=np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 3, 3, 1],
        'padding': 'VALID',
        'name': 'test_qint32_strides'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Strides > ksize
    input_dict_6 = {
        'input': np.arange(100, dtype=np.uint8).reshape(1, 10, 10, 1),
        'min_input': np.array(0.0, dtype=np.float32),
        'max_input': np.array(100.0, dtype=np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 3, 3, 1],
        'padding': 'SAME',
        'name': 'test_stride_gt_ksize'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 1x1 ksize (identity pooling)
    input_dict_7 = {
        'input': (np.arange(36, dtype=np.int8) - 18).reshape(1, 3, 3, 4),
        'min_input': np.array(-50.0, dtype=np.float32),
        'max_input': np.array(50.0, dtype=np.float32),
        'ksize': [1, 1, 1, 1],
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'test_1x1_ksize'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: ksize = input size (global pooling)
    input_dict_8 = {
        'input': np.arange(25, dtype=np.int32).reshape(1, 5, 5, 1),
        'min_input': np.array(-1e6, dtype=np.float32),
        'max_input': np.array(1e6, dtype=np.float32),
        'ksize': [1, 5, 5, 1],
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'test_global_pool_qint32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedMaxPool"] = tf_raw_ops_quantized_max_pool_inputs()

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
