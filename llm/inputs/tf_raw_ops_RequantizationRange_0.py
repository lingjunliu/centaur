
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_requantization_range_inputs():
    """
    This function generates a list of valid inputs for the
    tf.raw_ops.RequantizationRange operation.
    The error `InvalidArgumentError: Value for attr 'Tinput' of int8 is not in the list of allowed values: qint8...`
    confirms that the 'input' tensor must have a quantized dtype. The only way to
    create such a tensor is by using a TensorFlow quantization function.
    This code uses tf.quantization.quantize to produce tensors with the correct
    dtypes, which is the required fix for the reported error.
    """
    list_of_inputs = []

    def create_input_dict(name, float_values, min_range, max_range, q_type, **kwargs):
        """Helper to create quantized tensors and format the input dict."""
        float_tensor = tf.constant(float_values, dtype=tf.float32)
        quantized_input, _, _ = tf.quantization.quantize(
            float_tensor, min_range, max_range, T=q_type, **kwargs
        )
        return {
            'name': name,
            'input': quantized_input,
            'input_min': tf.constant(min_range, dtype=tf.float32),
            'input_max': tf.constant(max_range, dtype=tf.float32)
        }

    # Input 1: Correctly typed qint8 input to fix the InvalidArgumentError
    list_of_inputs.append(create_input_dict(
        'qint8_full_range_1d', [-1.0, 0.0, 1.0], -1.0, 1.0, tf.qint8
    ))

    # Input 2: Correctly typed quint8 input
    list_of_inputs.append(create_input_dict(
        'quint8_partial_range_1d', [50.0, 100.0, 150.0], 0.0, 255.0, tf.quint8
    ))

    # Input 3: Correctly typed 2D qint8 input
    list_of_inputs.append(create_input_dict(
        'qint8_2d_neg_range', [[-10.0, -7.5], [-5.0, -6.0]], -10.0, -5.0, tf.qint8
    ))

    # Input 4: Correctly typed qint32 input
    list_of_inputs.append(create_input_dict(
        'qint32_large_values', [-1000.0, 0.0, 1000.0], -1000.0, 1000.0, tf.qint32
    ))

    # Input 5: Correctly typed quint16 input
    list_of_inputs.append(create_input_dict(
        'quint16_1d', [0.0, 10.0, 300.0, 655.35], 0.0, 655.35, tf.quint16
    ))

    # Input 6: Correctly typed qint16 input
    list_of_inputs.append(create_input_dict(
        'qint16_3d', [[[-5.0, 0.0]], [[1.0, 5.0]]], -5.0, 5.0, tf.qint16
    ))

    # Input 7: qint8 with narrow_range=True
    list_of_inputs.append(create_input_dict(
        'qint8_narrow_range', [-1.0, 0.0, 1.0], -1.0, 1.0, tf.qint8, narrow_range=True
    ))

    # Input 8: quint8 with a single unique value
    list_of_inputs.append(create_input_dict(
        'quint8_single_value', [[0.5, 0.5], [0.5, 0.5]], 0.0, 1.0, tf.quint8
    ))

    # Input 9: qint8 with asymmetric float range
    list_of_inputs.append(create_input_dict(
        'qint8_asymmetric_range', [-5.0, 0.0, 1.0], -10.0, 2.0, tf.qint8
    ))
    
    # Input 10: quint8 where actual range is a subset of the quantization range
    list_of_inputs.append(create_input_dict(
        'quint8_actual_range_is_subset', [10.0, 20.0, 30.0, 40.0], 0.0, 255.0, tf.quint8
    ))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.RequantizationRange"] = tf_raw_ops_requantization_range_inputs()

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
