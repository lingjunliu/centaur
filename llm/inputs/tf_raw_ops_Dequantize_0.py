
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_dequantize_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.Dequantize function.
    The 'input' tensor for Dequantize must have a quantized dtype (e.g., tf.quint8).
    Standard integer dtypes like tf.uint8 are not accepted.
    We create the required input by bitcasting a standard integer tensor.
    """
    list_of_inputs = []

    # Case 1: MIN_COMBINED, quint8
    input_val = tf.bitcast(tf.constant([[0, 64], [128, 255]], dtype=tf.uint8), tf.quint8)
    list_of_inputs.append(
        {
            'input': input_val,
            'min_range': tf.constant(0.0, dtype=tf.float32),
            'max_range': tf.constant(6.0, dtype=tf.float32),
            'mode': 'MIN_COMBINED',
            'narrow_range': False,
            'axis': -1,
            'dtype': tf.float32,
            'name': 'min_combined_quint8'
        }
    )

    # Case 2: MIN_COMBINED, qint8
    input_val = tf.bitcast(tf.constant([-128, 0, 127], dtype=tf.int8), tf.qint8)
    list_of_inputs.append(
        {
            'input': input_val,
            'min_range': tf.constant(-1.0, dtype=tf.float32),
            'max_range': tf.constant(1.0, dtype=tf.float32),
            'mode': 'MIN_COMBINED',
            'narrow_range': False,
            'axis': -1,
            'dtype': tf.float32,
            'name': 'min_combined_qint8'
        }
    )

    # Case 3: MIN_FIRST, quint8
    input_val = tf.bitcast(tf.constant([[0, 128], [192, 255]], dtype=tf.uint8), tf.quint8)
    list_of_inputs.append(
        {
            'input': input_val,
            'min_range': tf.constant(-10.0, dtype=tf.float32),
            'max_range': tf.constant(10.0, dtype=tf.float32),
            'mode': 'MIN_FIRST',
            'narrow_range': False,
            'axis': -1,
            'dtype': tf.float32,
            'name': 'min_first_quint8'
        }
    )

    # Case 4: SCALED, quint8
    input_val = tf.bitcast(tf.constant([0, 51, 102, 153, 204, 255], dtype=tf.uint8), tf.quint8)
    list_of_inputs.append(
        {
            'input': input_val,
            'min_range': tf.constant(0.0, dtype=tf.float32),
            'max_range': tf.constant(100.0, dtype=tf.float32),
            'mode': 'SCALED',
            'narrow_range': False,
            'axis': -1,
            'dtype': tf.float32,
            'name': 'scaled_quint8'
        }
    )

    # Case 5: SCALED, qint8, narrow_range=True
    input_val = tf.bitcast(tf.constant([[-127, -64], [0, 64], [127, -1]], dtype=tf.int8), tf.qint8)
    list_of_inputs.append(
        {
            'input': input_val,
            'min_range': tf.constant(-5.0, dtype=tf.float32),
            'max_range': tf.constant(5.0, dtype=tf.float32),
            'mode': 'SCALED',
            'narrow_range': True,
            'axis': -1,
            'dtype': tf.float32,
            'name': 'scaled_qint8_narrow'
        }
    )

    # Case 6: qint16 input type
    input_val = tf.bitcast(tf.constant([-32768, 0, 32767], dtype=tf.int16), tf.qint16)
    list_of_inputs.append(
        {
            'input': input_val,
            'min_range': tf.constant(-1000.0, dtype=tf.float32),
            'max_range': tf.constant(1000.0, dtype=tf.float32),
            'mode': 'MIN_COMBINED',
            'narrow_range': False,
            'axis': -1,
            'dtype': tf.float32,
            'name': 'qint16_input'
        }
    )

    # Case 7: quint16 input type
    input_val = tf.bitcast(tf.constant([0, 10000, 65535], dtype=tf.uint16), tf.quint16)
    list_of_inputs.append(
        {
            'input': input_val,
            'min_range': tf.constant(0.0, dtype=tf.float32),
            'max_range': tf.constant(50000.0, dtype=tf.float32),
            'mode': 'MIN_FIRST',
            'narrow_range': False,
            'axis': -1,
            'dtype': tf.float32,
            'name': 'quint16_input'
        }
    )

    # Case 8: qint32 input type
    input_val = tf.bitcast(tf.constant(np.array([-2147483648, 0, 2147483647], dtype=np.int32)), tf.qint32)
    list_of_inputs.append(
        {
            'input': input_val,
            'min_range': tf.constant(-2.0e9, dtype=tf.float32),
            'max_range': tf.constant(2.0e9, dtype=tf.float32),
            'mode': 'MIN_COMBINED',
            'narrow_range': False,
            'axis': -1,
            'dtype': tf.float32,
            'name': 'qint32_input'
        }
    )

    # Case 9: bfloat16 output dtype
    input_val = tf.bitcast(tf.constant([[0, 255]], dtype=tf.uint8), tf.quint8)
    list_of_inputs.append(
        {
            'input': input_val,
            'min_range': tf.constant(0.0, dtype=tf.float32),
            'max_range': tf.constant(10.0, dtype=tf.float32),
            'mode': 'MIN_COMBINED',
            'narrow_range': False,
            'axis': -1,
            'dtype': tf.bfloat16,
            'name': 'bfloat16_output'
        }
    )

    # Case 10: Per-channel quantization, SCALED, qint8
    input_val = tf.bitcast(tf.constant(np.arange(12, dtype=np.int32).reshape((3, 4)) - 6, dtype=tf.int8), tf.qint8)
    list_of_inputs.append(
        {
            'input': input_val,
            'min_range': tf.constant([-1.0, -2.0, -3.0], dtype=tf.float32),
            'max_range': tf.constant([1.0, 2.0, 3.0], dtype=tf.float32),
            'mode': 'SCALED',
            'narrow_range': True,
            'axis': 0,
            'dtype': tf.float32,
            'name': 'per_channel_scaled_axis0'
        }
    )

    return list_of_inputs

generated_inputs["tf.raw_ops.Dequantize"] = tf_raw_ops_dequantize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Dequantize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Dequantize'.")

check_valid('tf.raw_ops.Dequantize', generated_inputs['tf.raw_ops.Dequantize'], lib="tf", suffix=0)
