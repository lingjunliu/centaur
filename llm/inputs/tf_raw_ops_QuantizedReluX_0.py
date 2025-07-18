
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_quantized_relu_x_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.QuantizedReluX.
    """
    list_of_inputs = []

    # Input 1: Basic quint8, 1D
    q_features_1, q_min_1, q_max_1 = tf.quantization.quantize(
        np.array([-10.0, 0.0, 5.0, 6.0, 10.0], dtype=np.float32), -10.0, 10.0, tf.quint8, mode='SCALED'
    )
    input_dict_1 = {
        'features': q_features_1.numpy(),
        'max_value': np.array(6.0, dtype=np.float32),
        'min_features': q_min_1.numpy(),
        'max_features': q_max_1.numpy(),
        'out_type': np.uint8,
        'name': 'basic_quint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: qint8 with negative values, 2D
    q_features_2, q_min_2, q_max_2 = tf.quantization.quantize(
        np.array([[-5.0, -2.5, 0.0], [2.5, 5.0, -0.1]], dtype=np.float32), -5.0, 5.0, tf.qint8, mode='SCALED'
    )
    input_dict_2 = {
        'features': q_features_2.numpy(),
        'max_value': np.array(3.0, dtype=np.float32),
        'min_features': q_min_2.numpy(),
        'max_features': q_max_2.numpy(),
        'out_type': np.int8,
        'name': 'qint8_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: qint16, 1D with large values
    q_features_3, q_min_3, q_max_3 = tf.quantization.quantize(
        np.array([-50.0, -10.0, 0.0, 10.0, 50.0], dtype=np.float32), -50.0, 50.0, tf.qint16, mode='SCALED'
    )
    input_dict_3 = {
        'features': q_features_3.numpy(),
        'max_value': np.array(20.0, dtype=np.float32),
        'min_features': q_min_3.numpy(),
        'max_features': q_max_3.numpy(),
        'out_type': np.int16,
        'name': 'qint16_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: quint16, 3D
    q_features_4, q_min_4, q_max_4 = tf.quantization.quantize(
        np.array([[[0., 10.], [20., 40.]], [[50., 200.], [1., 2.]]], dtype=np.float32), 0.0, 200.0, tf.quint16, mode='SCALED'
    )
    input_dict_4 = {
        'features': q_features_4.numpy(),
        'max_value': np.array(100.0, dtype=np.float32),
        'min_features': q_min_4.numpy(),
        'max_features': q_max_4.numpy(),
        'out_type': np.uint16,
        'name': 'quint16_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: qint32 with large range
    q_features_5, q_min_5, q_max_5 = tf.quantization.quantize(
        np.array([-2e6, 0., 2e6], dtype=np.float32), -2e6, 2e6, tf.qint32, mode='SCALED'
    )
    input_dict_5 = {
        'features': q_features_5.numpy(),
        'max_value': np.array(1e6, dtype=np.float32),
        'min_features': q_min_5.numpy(),
        'max_features': q_max_5.numpy(),
        'out_type': np.int32,
        'name': 'qint32_basic'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: max_value is very small, causing clipping
    q_features_6, q_min_6, q_max_6 = tf.quantization.quantize(
        np.array([0., 5., 10.], dtype=np.float32), 0.0, 10.0, tf.quint8, mode='SCALED'
    )
    input_dict_6 = {
        'features': q_features_6.numpy(),
        'max_value': np.array(0.5, dtype=np.float32),
        'min_features': q_min_6.numpy(),
        'max_features': q_max_6.numpy(),
        'out_type': np.uint8,
        'name': 'small_max_value'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: max_value is very large, larger than max_features
    q_features_7, q_min_7, q_max_7 = tf.quantization.quantize(
        np.array([[-5.0, 0.0, 5.0]], dtype=np.float32), -5.0, 5.0, tf.qint8, mode='SCALED'
    )
    input_dict_7 = {
        'features': q_features_7.numpy(),
        'max_value': np.array(1000.0, dtype=np.float32),
        'min_features': q_min_7.numpy(),
        'max_features': q_max_7.numpy(),
        'out_type': np.int8,
        'name': 'large_max_value'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Different out_type (qint8 in, quint16 out)
    q_features_8, q_min_8, q_max_8 = tf.quantization.quantize(
        np.array([-10., -1., 0., 5., 10.], dtype=np.float32), -10.0, 10.0, tf.qint8, mode='SCALED'
    )
    input_dict_8 = {
        'features': q_features_8.numpy(),
        'max_value': np.array(8.0, dtype=np.float32),
        'min_features': q_min_8.numpy(),
        'max_features': q_max_8.numpy(),
        'out_type': np.uint16,
        'name': 'different_out_type'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Scalar feature tensor
    q_features_9, q_min_9, q_max_9 = tf.quantization.quantize(
        np.array(5.0, dtype=np.float32), 0.0, 10.0, tf.quint8, mode='SCALED'
    )
    input_dict_9 = {
        'features': q_features_9.numpy(),
        'max_value': np.array(5.0, dtype=np.float32),
        'min_features': q_min_9.numpy(),
        'max_features': q_max_9.numpy(),
        'out_type': np.uint8,
        'name': 'scalar_feature'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: min_features equals max_features (zero range)
    q_features_10, q_min_10, q_max_10 = tf.quantization.quantize(
        np.array([2.0, 2.0, 2.0], dtype=np.float32), 2.0, 2.0, tf.qint8, mode='SCALED'
    )
    input_dict_10 = {
        'features': q_features_10.numpy(),
        'max_value': np.array(5.0, dtype=np.float32),
        'min_features': q_min_10.numpy(),
        'max_features': q_max_10.numpy(),
        'out_type': np.int8,
        'name': 'zero_range'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedReluX"] = tf_raw_ops_quantized_relu_x_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedReluX' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedReluX'.")

check_valid('tf.raw_ops.QuantizedReluX', generated_inputs['tf.raw_ops.QuantizedReluX'], lib="tf", suffix=0)
