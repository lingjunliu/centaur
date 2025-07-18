
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_quantization_dequantize_inputs():
    """
    Generates a list of inputs for the tf.quantization.dequantize function.
    """
    list_of_inputs = []

    # Input 1: Basic quint8, MIN_COMBINED mode.
    input_dict_1 = {
        'input': np.array([0, 128, 255], dtype=np.uint8),
        'min_range': np.array(0.0, dtype=np.float32),
        'max_range': np.array(6.0, dtype=np.float32),
        'mode': 'MIN_COMBINED',
        'name': 'quint8_min_combined',
        'axis': -1,
        'narrow_range': False,
        'dtype': tf.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic qint8, MIN_COMBINED mode.
    input_dict_2 = {
        'input': np.array([-128, 0, 127], dtype=np.int8),
        'min_range': np.array(-10.0, dtype=np.float32),
        'max_range': np.array(10.0, dtype=np.float32),
        'mode': 'MIN_COMBINED',
        'name': 'qint8_min_combined',
        'axis': -1,
        'narrow_range': False,
        'dtype': tf.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: MIN_FIRST mode with quint8.
    input_dict_3 = {
        'input': np.array([[0, 64], [192, 255]], dtype=np.uint8),
        'min_range': np.array(-1.0, dtype=np.float32),
        'max_range': np.array(1.0, dtype=np.float32),
        'mode': 'MIN_FIRST',
        'name': 'quint8_min_first',
        'axis': -1,
        'narrow_range': False,
        'dtype': tf.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: SCALED mode with qint8.
    input_dict_4 = {
        'input': np.array([-128, -64, 0, 64, 127], dtype=np.int8),
        'min_range': np.array(-1.0, dtype=np.float32),
        'max_range': np.array(1.0, dtype=np.float32),
        'mode': 'SCALED',
        'name': 'qint8_scaled',
        'axis': -1,
        'narrow_range': False,
        'dtype': tf.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: SCALED mode with narrow_range=True.
    input_dict_5 = {
        'input': np.array([-127, 0, 127], dtype=np.int8),
        'min_range': np.array(-5.0, dtype=np.float32),
        'max_range': np.array(5.0, dtype=np.float32),
        'mode': 'SCALED',
        'name': 'qint8_scaled_narrow',
        'axis': -1,
        'narrow_range': True,
        'dtype': tf.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Per-axis quantization (axis=0).
    input_dict_6 = {
        'input': np.array([[0, 128, 255], [0, 64, 128]], dtype=np.uint8),
        'min_range': np.array([0.0, -10.0], dtype=np.float32),
        'max_range': np.array([1.0, 0.0], dtype=np.float32),
        'mode': 'MIN_COMBINED',
        'name': 'per_axis_quant_axis0',
        'axis': 0,
        'narrow_range': False,
        'dtype': tf.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: qint32 input type.
    input_dict_7 = {
        'input': np.array([np.iinfo(np.int32).min, 0, np.iinfo(np.int32).max], dtype=np.int32),
        'min_range': np.array(-1000.0, dtype=np.float32),
        'max_range': np.array(1000.0, dtype=np.float32),
        'mode': 'MIN_COMBINED',
        'name': 'qint32_input',
        'axis': -1,
        'narrow_range': False,
        'dtype': tf.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: bfloat16 output dtype.
    input_dict_8 = {
        'input': np.array([0, 1, 254, 255], dtype=np.uint8),
        'min_range': np.array(0.0, dtype=np.float32),
        'max_range': np.array(1.0, dtype=np.float32),
        'mode': 'MIN_COMBINED',
        'name': 'bfloat16_output',
        'axis': -1,
        'narrow_range': False,
        'dtype': tf.bfloat16
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: qint16 input type.
    input_dict_9 = {
        'input': np.array([np.iinfo(np.int16).min, 0, np.iinfo(np.int16).max], dtype=np.int16),
        'min_range': np.array(-1.0, dtype=np.float32),
        'max_range': np.array(1.0, dtype=np.float32),
        'mode': 'MIN_COMBINED',
        'name': 'qint16_input',
        'axis': -1,
        'narrow_range': False,
        'dtype': tf.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: quint16 input type.
    input_dict_10 = {
        'input': np.array([np.iinfo(np.uint16).min, 32768, np.iinfo(np.uint16).max], dtype=np.uint16),
        'min_range': np.array(0.0, dtype=np.float32),
        'max_range': np.array(100.0, dtype=np.float32),
        'mode': 'MIN_COMBINED',
        'name': 'quint16_input',
        'axis': -1,
        'narrow_range': False,
        'dtype': tf.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.quantization.dequantize"] = get_tf_quantization_dequantize_inputs()

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
