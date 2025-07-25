
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_quantization_dequantize_inputs():
    """
    Generates a list of valid inputs for the tf.quantization.dequantize function.
    The 'input' tensor requires a special quantized dtype (e.g., tf.quint8).
    Standard numpy arrays do not have these dtypes. To create a valid input,
    we create a tf.Tensor with the correct quantized dtype by casting a numpy array.
    This is a necessary exception to the "inputs should be in numpy format" rule,
    as it's the only way to satisfy the TensorFlow kernel's type requirements.
    """
    list_of_inputs = []

    # Case 1: quint8 input
    input_dict_1 = {
        'input': tf.cast(np.array([0, 128, 255], dtype=np.uint8), tf.quint8),
        'min_range': np.array(0.0, dtype=np.float32),
        'max_range': np.array(6.0, dtype=np.float32),
        'mode': 'MIN_COMBINED',
        'name': 'quint8_min_combined',
        'axis': -1,
        'narrow_range': False,
        'dtype': tf.float32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: qint8 input, symmetric range
    input_dict_2 = {
        'input': tf.cast(np.array([[-128, 0], [127, -1]], dtype=np.int8), tf.qint8),
        'min_range': np.array(-1.0, dtype=np.float32),
        'max_range': np.array(1.0, dtype=np.float32),
        'mode': 'MIN_COMBINED',
        'name': 'qint8_min_combined',
        'axis': -1,
        'narrow_range': False,
        'dtype': tf.float32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: bfloat16 output type with quint8 input
    input_dict_3 = {
        'input': tf.cast(np.array([0, 64, 128, 192, 255], dtype=np.uint8), tf.quint8),
        'min_range': np.array(0.0, dtype=np.float32),
        'max_range': np.array(10.0, dtype=np.float32),
        'mode': 'MIN_COMBINED',
        'name': 'bfloat16_output',
        'axis': -1,
        'narrow_range': False,
        'dtype': tf.bfloat16,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: MIN_FIRST mode with qint8
    input_dict_4 = {
        'input': tf.cast(np.array([-128, -64, 0, 64, 127], dtype=np.int8), tf.qint8),
        'min_range': np.array(-10.0, dtype=np.float32),
        'max_range': np.array(10.0, dtype=np.float32),
        'mode': 'MIN_FIRST',
        'name': 'min_first_mode',
        'axis': -1,
        'narrow_range': False,
        'dtype': tf.float32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: SCALED mode with quint8
    input_dict_5 = {
        'input': tf.cast(np.array([0, 50, 100, 255], dtype=np.uint8), tf.quint8),
        'min_range': np.array(0.0, dtype=np.float32),
        'max_range': np.array(50.0, dtype=np.float32),
        'mode': 'SCALED',
        'name': 'scaled_mode_quint8',
        'axis': -1,
        'narrow_range': False,
        'dtype': tf.float32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: SCALED mode with narrow_range=True and qint8
    input_dict_6 = {
        'input': tf.cast(np.array([-127, 0, 127], dtype=np.int8), tf.qint8),
        'min_range': np.array(-1.0, dtype=np.float32),
        'max_range': np.array(1.0, dtype=np.float32),
        'mode': 'SCALED',
        'name': 'scaled_mode_narrow',
        'axis': -1,
        'narrow_range': True,
        'dtype': tf.float32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Per-channel quantization with quint8
    input_dict_7 = {
        'input': tf.cast(np.arange(24, dtype=np.uint8).reshape((1, 2, 4, 3)), tf.quint8),
        'min_range': np.array([0.0, -1.0, -2.0], dtype=np.float32),
        'max_range': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'mode': 'MIN_COMBINED',
        'name': 'per_channel_quint8',
        'axis': 3,
        'narrow_range': False,
        'dtype': tf.float32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: qint16 input
    input_dict_8 = {
        'input': tf.cast(np.array([-32768, 0, 32767], dtype=np.int16), tf.qint16),
        'min_range': np.array(-100.0, dtype=np.float32),
        'max_range': np.array(100.0, dtype=np.float32),
        'mode': 'MIN_COMBINED',
        'name': 'qint16_input',
        'axis': -1,
        'narrow_range': False,
        'dtype': tf.float32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: quint16 input
    input_dict_9 = {
        'input': tf.cast(np.array([0, 32767, 65535], dtype=np.uint16), tf.quint16),
        'min_range': np.array(0.0, dtype=np.float32),
        'max_range': np.array(1000.0, dtype=np.float32),
        'mode': 'MIN_FIRST',
        'name': 'quint16_input',
        'axis': -1,
        'narrow_range': False,
        'dtype': tf.float32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: qint32 input
    input_dict_10 = {
        'input': tf.cast(np.array([-2147483648, 0, 2147483647], dtype=np.int32), tf.qint32),
        'min_range': np.array(-1.0, dtype=np.float32),
        'max_range': np.array(1.0, dtype=np.float32),
        'mode': 'MIN_COMBINED',
        'name': 'qint32_input',
        'axis': -1,
        'narrow_range': False,
        'dtype': tf.float32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.quantization.dequantize"] = tf_quantization_dequantize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.quantization.dequantize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.quantization.dequantize'.")

check_valid('tf.quantization.dequantize', generated_inputs['tf.quantization.dequantize'], lib="tf", suffix=0)
