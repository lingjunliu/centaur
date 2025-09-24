
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_quantizedinstancenorm_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.QuantizedInstanceNorm operation.
    The previous errors indicate that providing standard numpy integer arrays results
    in tf.int* tensors, not the required tf.qint* tensors. The only way to create
    a tensor with a quantized dtype is to use a TensorFlow quantization function.
    This implementation creates valid tf.q* tensors for the 'x' input, resolving the error.
    """
    list_of_inputs = []

    def get_quantized_tensor(shape, min_val, max_val, q_type):
        """Generates a quantized tensor and its float range."""
        float_x = np.random.uniform(min_val, max_val, size=shape).astype(np.float32)
        # tf.quantization.quantize creates the quantized tensor and returns its effective range
        q_x, q_min, q_max = tf.quantization.quantize(float_x, min_val, max_val, T=q_type, mode='SCALED')
        # The op expects float32 tensors for min and max, not python floats.
        return q_x, tf.constant(q_min, dtype=tf.float32), tf.constant(q_max, dtype=tf.float32)

    # Input 1: Basic case with qint8
    x, x_min, x_max = get_quantized_tensor((1, 4, 4, 3), -1.0, 1.0, tf.qint8)
    input_dict = {
        'x': x,
        'x_min': x_min,
        'x_max': x_max,
        'output_range_given': False,
        'given_y_min': 0.0,
        'given_y_max': 0.0,
        'variance_epsilon': 1e-05,
        'min_separation': 0.001,
        'name': 'test_qint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic case with quint8 and given output range
    x, x_min, x_max = get_quantized_tensor((2, 5, 5, 2), 0.0, 10.0, tf.quint8)
    input_dict = {
        'x': x,
        'x_min': x_min,
        'x_max': x_max,
        'output_range_given': True,
        'given_y_min': 0.0,
        'given_y_max': 5.0,
        'variance_epsilon': 1e-05,
        'min_separation': 0.001,
        'name': 'test_quint8_given_range'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: qint16 with custom epsilon
    x, x_min, x_max = get_quantized_tensor((1, 3, 3, 5), -100.0, 100.0, tf.qint16)
    input_dict = {
        'x': x,
        'x_min': x_min,
        'x_max': x_max,
        'output_range_given': False,
        'given_y_min': 0.0,
        'given_y_max': 0.0,
        'variance_epsilon': 1e-04,
        'min_separation': 0.001,
        'name': 'test_qint16_custom_epsilon'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: quint16 with custom min_separation
    x, x_min, x_max = get_quantized_tensor((1, 2, 2, 8), 0.0, 1000.0, tf.quint16)
    input_dict = {
        'x': x,
        'x_min': x_min,
        'x_max': x_max,
        'output_range_given': False,
        'given_y_min': 0.0,
        'given_y_max': 0.0,
        'variance_epsilon': 1e-05,
        'min_separation': 0.1,
        'name': 'test_quint16_custom_sep'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: qint32 with large range and given output
    x, x_min, x_max = get_quantized_tensor((1, 2, 2, 2), -50000.0, 50000.0, tf.qint32)
    input_dict = {
        'x': x,
        'x_min': x_min,
        'x_max': x_max,
        'output_range_given': True,
        'given_y_min': -100.0,
        'given_y_max': 100.0,
        'variance_epsilon': 1e-05,
        'min_separation': 0.001,
        'name': 'test_qint32_large_range'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Larger batch size with qint8
    x, x_min, x_max = get_quantized_tensor((4, 3, 3, 3), -10.0, 10.0, tf.qint8)
    input_dict = {
        'x': x,
        'x_min': x_min,
        'x_max': x_max,
        'output_range_given': False,
        'given_y_min': 0.0,
        'given_y_max': 0.0,
        'variance_epsilon': 1e-05,
        'min_separation': 0.001,
        'name': 'test_qint8_large_batch'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative range for given output
    x, x_min, x_max = get_quantized_tensor((1, 6, 6, 1), -5.0, 5.0, tf.qint8)
    input_dict = {
        'x': x,
        'x_min': x_min,
        'x_max': x_max,
        'output_range_given': True,
        'given_y_min': -2.0,
        'given_y_max': -0.5,
        'variance_epsilon': 1e-06,
        'min_separation': 0.01,
        'name': 'test_qint8_neg_output'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: quint8 with 1x1 spatial dimensions
    x, x_min, x_max = get_quantized_tensor((3, 1, 1, 10), 0.0, 25.5, tf.quint8)
    input_dict = {
        'x': x,
        'x_min': x_min,
        'x_max': x_max,
        'output_range_given': False,
        'given_y_min': 0.0,
        'given_y_max': 0.0,
        'variance_epsilon': 1e-05,
        'min_separation': 0.001,
        'name': 'test_quint8_1x1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: qint16 with all parameters specified
    x, x_min, x_max = get_quantized_tensor((2, 4, 4, 2), -256.0, 256.0, tf.qint16)
    input_dict = {
        'x': x,
        'x_min': x_min,
        'x_max': x_max,
        'output_range_given': True,
        'given_y_min': -12.0,
        'given_y_max': 12.0,
        'variance_epsilon': 1e-03,
        'min_separation': 0.5,
        'name': 'test_qint16_all_params'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: qint32 with default parameters
    x, x_min, x_max = get_quantized_tensor((1, 2, 2, 1), -1.0, 1.0, tf.qint32)
    input_dict = {
        'x': x,
        'x_min': x_min,
        'x_max': x_max,
        'output_range_given': False,
        'given_y_min': 0.0,
        'given_y_max': 0.0,
        'variance_epsilon': 1e-05,
        'min_separation': 0.001,
        'name': 'test_qint32_default'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedInstanceNorm"] = tf_raw_ops_quantizedinstancenorm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedInstanceNorm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedInstanceNorm'.")

check_valid('tf.raw_ops.QuantizedInstanceNorm', generated_inputs['tf.raw_ops.QuantizedInstanceNorm'], lib="tf", suffix=0)
