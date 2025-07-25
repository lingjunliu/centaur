
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_requantize_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.Requantize.
    NOTE: This operation requires the input tensor to have a quantized dtype (e.g., tf.qint16),
    which cannot be represented by a standard NumPy array. Therefore, to create valid inputs,
    we must generate `tf.Tensor` objects with the correct quantized dtypes, which deviates
    from the "numpy format" constraint. This is the only way to resolve the InvalidArgumentError.
    """
    list_of_inputs = []

    # Input 1: qint16 -> qint8
    quant_input, quant_min, quant_max = tf.quantization.quantize(
        tf.constant([[-1.0, 0.0], [0.5, 1.0]], dtype=tf.float32), -1.0, 1.0, T=tf.qint16)
    list_of_inputs.append({
        'name': 'qint16_to_qint8',
        'input': quant_input,
        'input_min': quant_min,
        'input_max': quant_max,
        'requested_output_min': np.array(-1.0, dtype=np.float32),
        'requested_output_max': np.array(1.0, dtype=np.float32),
        'out_type': tf.qint8,
    })

    # Input 2: quint16 -> quint8
    quant_input, quant_min, quant_max = tf.quantization.quantize(
        tf.constant([[0., 100.], [128., 255.]], dtype=tf.float32), 0.0, 255.0, T=tf.quint16)
    list_of_inputs.append({
        'name': 'quint16_to_quint8',
        'input': quant_input,
        'input_min': quant_min,
        'input_max': quant_max,
        'requested_output_min': np.array(0.0, dtype=np.float32),
        'requested_output_max': np.array(255.0, dtype=np.float32),
        'out_type': tf.quint8,
    })

    # Input 3: qint32 -> qint8
    quant_input, quant_min, quant_max = tf.quantization.quantize(
        tf.constant([0, 0.5, 1.0, -1.0], dtype=tf.float32), -1.0, 1.0, T=tf.qint32)
    list_of_inputs.append({
        'name': 'qint32_to_qint8',
        'input': quant_input,
        'input_min': quant_min,
        'input_max': quant_max,
        'requested_output_min': np.array(-1.0, dtype=np.float32),
        'requested_output_max': np.array(1.0, dtype=np.float32),
        'out_type': tf.qint8,
    })

    # Input 4: qint32 -> qint16
    quant_input, quant_min, quant_max = tf.quantization.quantize(
        tf.constant([-1000., 0., 1000.], dtype=tf.float32), -1000.0, 1000.0, T=tf.qint32)
    list_of_inputs.append({
        'name': 'qint32_to_qint16_wide_range',
        'input': quant_input,
        'input_min': quant_min,
        'input_max': quant_max,
        'requested_output_min': np.array(-1000.0, dtype=np.float32),
        'requested_output_max': np.array(1000.0, dtype=np.float32),
        'out_type': tf.qint16,
    })

    # Input 5: qint16 to qint8 with clipping
    quant_input, quant_min, quant_max = tf.quantization.quantize(
        tf.constant([-1.0, -0.5, 0.0, 0.6, 1.0], dtype=tf.float32), -1.0, 1.0, T=tf.qint16)
    list_of_inputs.append({
        'name': 'qint16_to_qint8_clipping',
        'input': quant_input,
        'input_min': quant_min,
        'input_max': quant_max,
        'requested_output_min': np.array(0.0, dtype=np.float32),
        'requested_output_max': np.array(0.5, dtype=np.float32),
        'out_type': tf.qint8,
    })
    
    # Input 6: qint32 to quint16 with asymmetric range
    quant_input, quant_min, quant_max = tf.quantization.quantize(
        tf.constant([-3., 0., 1.], dtype=tf.float32), -3.0, 1.0, T=tf.qint32)
    list_of_inputs.append({
        'name': 'qint32_to_quint16_asymmetric',
        'input': quant_input,
        'input_min': quant_min,
        'input_max': quant_max,
        'requested_output_min': np.array(0.0, dtype=np.float32),
        'requested_output_max': np.array(4.0, dtype=np.float32),
        'out_type': tf.quint16,
    })

    # Input 7: quint16 to qint8
    quant_input, quant_min, quant_max = tf.quantization.quantize(
        tf.constant([[0., 1.], [2., 4.]], dtype=tf.float32), 0.0, 4.0, T=tf.quint16)
    list_of_inputs.append({
        'name': 'quint16_to_qint8',
        'input': quant_input,
        'input_min': quant_min,
        'input_max': quant_max,
        'requested_output_min': np.array(-2.0, dtype=np.float32),
        'requested_output_max': np.array(2.0, dtype=np.float32),
        'out_type': tf.qint8,
    })

    # Input 8: Empty input tensor
    quant_input, quant_min, quant_max = tf.quantization.quantize(
        tf.constant([], dtype=tf.float32), -1.0, 1.0, T=tf.qint16)
    list_of_inputs.append({
        'name': 'empty_input_qint16_to_qint8',
        'input': quant_input,
        'input_min': quant_min,
        'input_max': quant_max,
        'requested_output_min': np.array(-1.0, dtype=np.float32),
        'requested_output_max': np.array(1.0, dtype=np.float32),
        'out_type': tf.qint8,
    })

    # Input 9: Zero-scaling (input_max == input_min)
    quant_input, quant_min, quant_max = tf.quantization.quantize(
        tf.constant([[5.0, 5.0], [5.0, 5.0]], dtype=tf.float32), 5.0, 5.0, T=tf.qint16)
    list_of_inputs.append({
        'name': 'zero_scaling_qint16_to_qint8',
        'input': quant_input,
        'input_min': quant_min,
        'input_max': quant_max,
        'requested_output_min': np.array(-1.0, dtype=np.float32),
        'requested_output_max': np.array(1.0, dtype=np.float32),
        'out_type': tf.qint8,
    })

    # Input 10: 3D tensor input
    quant_input, quant_min, quant_max = tf.quantization.quantize(
        tf.reshape(tf.range(-8., 8., 1.), (2, 2, 4)), -10.0, 10.0, T=tf.qint32)
    list_of_inputs.append({
        'name': 'qint32_to_qint8_3d',
        'input': quant_input,
        'input_min': quant_min,
        'input_max': quant_max,
        'requested_output_min': np.array(-5.0, dtype=np.float32),
        'requested_output_max': np.array(5.0, dtype=np.float32),
        'out_type': tf.qint8,
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.Requantize"] = tf_raw_ops_requantize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Requantize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Requantize'.")

check_valid('tf.raw_ops.Requantize', generated_inputs['tf.raw_ops.Requantize'], lib="tf", suffix=0)
