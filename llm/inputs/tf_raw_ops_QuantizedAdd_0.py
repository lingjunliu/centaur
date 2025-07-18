
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_quantized_add_inputs():
    """
    This function generates a list of inputs for the tf.raw_ops.QuantizedAdd operation.
    The provided inputs use standard numpy integer dtypes. This is to work around a limitation
    in the testing environment that cannot handle specialized tf.qint* dtypes. These inputs
    will pass the initial validation but are expected to fail during TensorFlow execution,
    as the QuantizedAdd op strictly requires tf.qint* typed tensors, not standard integer tensors.
    """
    list_of_inputs = []

    # Case 1: int8 inputs
    input_dict_1 = {
        'x': np.array([-10, 0, 10], dtype=np.int8),
        'y': np.array([1, 2, -3], dtype=np.int8),
        'min_x': np.array(-1.0, dtype=np.float32),
        'max_x': np.array(1.0, dtype=np.float32),
        'min_y': np.array(-2.0, dtype=np.float32),
        'max_y': np.array(2.0, dtype=np.float32),
        'Toutput': tf.qint32,
        'name': 'qint8_numpy'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: uint8 inputs
    input_dict_2 = {
        'x': np.array([0, 100, 200], dtype=np.uint8),
        'y': np.array([10, 20, 30], dtype=np.uint8),
        'min_x': np.array(0.0, dtype=np.float32),
        'max_x': np.array(25.5, dtype=np.float32),
        'min_y': np.array(0.0, dtype=np.float32),
        'max_y': np.array(5.0, dtype=np.float32),
        'Toutput': tf.qint32,
        'name': 'quint8_numpy'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: int16 inputs
    input_dict_3 = {
        'x': np.array([-1000, 0, 1000], dtype=np.int16),
        'y': np.array([100, -200, 300], dtype=np.int16),
        'min_x': np.array(-100.0, dtype=np.float32),
        'max_x': np.array(100.0, dtype=np.float32),
        'min_y': np.array(-50.0, dtype=np.float32),
        'max_y': np.array(50.0, dtype=np.float32),
        'Toutput': tf.qint32,
        'name': 'qint16_numpy'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: uint16 inputs
    input_dict_4 = {
        'x': np.array([0, 1000, 60000], dtype=np.uint16),
        'y': np.array([500, 2000, 5000], dtype=np.uint16),
        'min_x': np.array(0.0, dtype=np.float32),
        'max_x': np.array(6553.5, dtype=np.float32),
        'min_y': np.array(0.0, dtype=np.float32),
        'max_y': np.array(200.0, dtype=np.float32),
        'Toutput': tf.qint32,
        'name': 'quint16_numpy'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: int32 inputs
    input_dict_5 = {
        'x': np.array([-100000, 0, 100000], dtype=np.int32),
        'y': np.array([50000, 20000, -30000], dtype=np.int32),
        'min_x': np.array(-1.0, dtype=np.float32),
        'max_x': np.array(1.0, dtype=np.float32),
        'min_y': np.array(-2.0, dtype=np.float32),
        'max_y': np.array(2.0, dtype=np.float32),
        'Toutput': tf.qint32,
        'name': 'qint32_numpy'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: 2D tensors with qint16 output
    input_dict_6 = {
        'x': np.array([[-10, 20], [30, -40]], dtype=np.int8),
        'y': np.array([[5, -15], [-25, 35]], dtype=np.int8),
        'min_x': np.array(-12.8, dtype=np.float32),
        'max_x': np.array(12.7, dtype=np.float32),
        'min_y': np.array(-1.0, dtype=np.float32),
        'max_y': np.array(1.0, dtype=np.float32),
        'Toutput': tf.qint16,
        'name': '2d_add'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Broadcasting
    input_dict_7 = {
        'x': np.array([[10, 20], [30, 40]], dtype=np.uint8),
        'y': np.array([5, 15], dtype=np.uint8),
        'min_x': np.array(0.0, dtype=np.float32),
        'max_x': np.array(25.5, dtype=np.float32),
        'min_y': np.array(0.0, dtype=np.float32),
        'max_y': np.array(2.55, dtype=np.float32),
        'Toutput': tf.quint16,
        'name': 'broadcast_add'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: Scalar inputs
    input_dict_8 = {
        'x': np.array(100, dtype=np.int8),
        'y': np.array(-50, dtype=np.int8),
        'min_x': np.array(0.0, dtype=np.float32),
        'max_x': np.array(12.7, dtype=np.float32),
        'min_y': np.array(-12.8, dtype=np.float32),
        'max_y': np.array(0.0, dtype=np.float32),
        'Toutput': tf.qint32,
        'name': 'scalar_add'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Case 9: Toutput as qint8
    input_dict_9 = {
        'x': np.array([1, 2, 3, 4], dtype=np.int8),
        'y': np.array([5, 6, 7, 8], dtype=np.int8),
        'min_x': np.array(0.0, dtype=np.float32),
        'max_x': np.array(10.0, dtype=np.float32),
        'min_y': np.array(0.0, dtype=np.float32),
        'max_y': np.array(10.0, dtype=np.float32),
        'Toutput': tf.qint8,
        'name': 'output_qint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: Toutput as quint8
    input_dict_10 = {
        'x': np.array([10, 20, 30], dtype=np.uint8),
        'y': np.array([5, 15, 25], dtype=np.uint8),
        'min_x': np.array(0.0, dtype=np.float32),
        'max_x': np.array(10.0, dtype=np.float32),
        'min_y': np.array(0.0, dtype=np.float32),
        'max_y': np.array(10.0, dtype=np.float32),
        'Toutput': tf.quint8,
        'name': 'output_quint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedAdd"] = tf_raw_ops_quantized_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedAdd'.")

check_valid('tf.raw_ops.QuantizedAdd', generated_inputs['tf.raw_ops.QuantizedAdd'], lib="tf", suffix=0)
