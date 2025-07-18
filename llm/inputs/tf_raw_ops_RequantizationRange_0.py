
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_requantization_range_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.RequantizationRange.
    """
    list_of_inputs = []

    # Case 1: Corresponds to qint8
    input_dict_1 = {
        'name': 'test_qint8',
        'input': np.array([-128, 0, 127], dtype=np.int8),
        'input_min': np.array(-1.0, dtype=np.float32),
        'input_max': np.array(1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: Corresponds to quint8
    input_dict_2 = {
        'name': 'test_quint8',
        'input': np.array([[10, 50], [100, 200]], dtype=np.uint8),
        'input_min': np.array(0.0, dtype=np.float32),
        'input_max': np.array(25.5, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Corresponds to qint32
    input_dict_3 = {
        'name': 'test_qint32',
        'input': np.array([[[100000, -200000], [0, 50000]]], dtype=np.int32),
        'input_min': np.array(-1000.0, dtype=np.float32),
        'input_max': np.array(1000.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Corresponds to qint16
    input_dict_4 = {
        'name': 'test_qint16',
        'input': np.array([5000], dtype=np.int16),
        'input_min': np.array(-32768.0, dtype=np.float32),
        'input_max': np.array(32767.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Corresponds to quint16
    input_dict_5 = {
        'name': 'test_quint16',
        'input': np.array([[0, 10000], [30000, 65535]], dtype=np.uint16),
        'input_min': np.array(0.0, dtype=np.float32),
        'input_max': np.array(6553.5, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Corresponds to qint8, single repeated value
    input_dict_6 = {
        'name': 'test_qint8_repeated',
        'input': np.array([[50, 50], [50, 50]], dtype=np.int8),
        'input_min': np.array(-10.0, dtype=np.float32),
        'input_max': np.array(10.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Corresponds to qint16, negative float range
    input_dict_7 = {
        'name': 'test_qint16_neg_range',
        'input': np.array([[[-100, 0]], [[100, 200]]], dtype=np.int16),
        'input_min': np.array(-50.0, dtype=np.float32),
        'input_max': np.array(-1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: Corresponds to quint8, zero float range
    input_dict_8 = {
        'name': 'test_quint8_zero_range',
        'input': np.array([10, 20, 30, 40], dtype=np.uint8),
        'input_min': np.array(0.0, dtype=np.float32),
        'input_max': np.array(0.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Case 9: Corresponds to qint32, large range
    input_dict_9 = {
        'name': 'test_qint32_large_range',
        'input': np.array([-1000000, 0, 1000000], dtype=np.int32),
        'input_min': np.array(-2.1e9, dtype=np.float32),
        'input_max': np.array(2.1e9, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: Corresponds to quint16, 4D tensor
    input_dict_10 = {
        'name': 'test_quint16_4d',
        'input': np.arange(16, dtype=np.uint16).reshape((2, 2, 2, 2)),
        'input_min': np.array(0.1, dtype=np.float32),
        'input_max': np.array(0.9, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.RequantizationRange"] = get_requantization_range_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RequantizationRange' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RequantizationRange'.")

check_valid('tf.raw_ops.RequantizationRange', generated_inputs['tf.raw_ops.RequantizationRange'], lib="tf", suffix=0)
