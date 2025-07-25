
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_quantized_relu_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.QuantizedRelu function.
    Uses tf.quantization.quantize to create correctly typed quantized tensors.
    """
    list_of_inputs = []

    def create_quantized_input(float_features, min_val, max_val, quant_type, out_type, name):
        """Helper to create a single valid input dictionary."""
        # This is the correct way to create the required quantized tensor type
        features_tensor, min_features_tensor, max_features_tensor = tf.quantization.quantize(
            tf.constant(float_features, dtype=tf.float32), min_val, max_val, T=quant_type
        )
        return {
            'features': features_tensor,
            'min_features': min_features_tensor,
            'max_features': max_features_tensor,
            'out_type': out_type,
            'name': name
        }

    # Case 1: Basic qint8 input
    list_of_inputs.append(create_quantized_input(
        [-10.0, -5.0, 0.0, 5.0, 9.9], -10.0, 10.0, tf.qint8, tf.qint8, 'qint8_basic'
    ))

    # Case 2: Basic quint8 input
    list_of_inputs.append(create_quantized_input(
        [[0.0, 1.5], [12.8, 25.5]], 0.0, 25.5, tf.quint8, tf.quint8, 'quint8_basic'
    ))

    # Case 3: Basic qint32 input
    list_of_inputs.append(create_quantized_input(
        [[[-200.0, -1.0], [0.0, 100.0]]], -200.0, 100.0, tf.qint32, tf.qint32, 'qint32_basic'
    ))

    # Case 4: Basic qint16 input
    list_of_inputs.append(create_quantized_input(
        [-5.0, -0.1, 0.0, 1.0, 4.9], -5.0, 5.0, tf.qint16, tf.qint16, 'qint16_basic'
    ))

    # Case 5: Basic quint16 input
    list_of_inputs.append(create_quantized_input(
        [[0.0, 100.0], [300.0, 655.3]], 0.0, 655.35, tf.quint16, tf.quint16, 'quint16_basic'
    ))

    # Case 6: Input qint8, output quint8 (type change)
    list_of_inputs.append(create_quantized_input(
        [-1.0, -0.5, 0.0, 0.5, 1.0], -1.0, 1.0, tf.qint8, tf.quint8, 'qint8_to_quint8'
    ))

    # Case 7: All negative feature values
    list_of_inputs.append(create_quantized_input(
        [[-0.1, -1.0], [-5.0, -12.8]], -12.8, -0.01, tf.qint8, tf.qint8, 'all_negative_features'
    ))

    # Case 8: Input qint32, output qint8 (down-casting bit depth)
    list_of_inputs.append(create_quantized_input(
        [-200.0, 0.0, 200.0], -200.0, 200.0, tf.qint32, tf.qint8, 'qint32_to_qint8'
    ))

    # Case 9: Empty features tensor
    list_of_inputs.append(create_quantized_input(
        [], -1.0, 1.0, tf.qint8, tf.qint8, 'empty_features'
    ))

    # Case 10: All positive quint8 features
    list_of_inputs.append(create_quantized_input(
        [0.1, 1.5, 3.0], 0.0, 3.0, tf.quint8, tf.quint8, 'all_positive_quint8'
    ))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedRelu"] = get_quantized_relu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedRelu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedRelu'.")

check_valid('tf.raw_ops.QuantizedRelu', generated_inputs['tf.raw_ops.QuantizedRelu'], lib="tf", suffix=0)
