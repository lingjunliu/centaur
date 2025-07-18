
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_quantizedbiasadd_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.QuantizedBiasAdd function.
    The `tf.raw_ops.QuantizedBiasAdd` operation requires input tensors of a
    quantized type (e.g., tf.qint8), not a standard integer type (e.g., tf.int8).
    Passing standard numpy arrays results in tf.int8 tensors, causing an
    `InvalidArgumentError`.
    The only correct way to generate a tensor with a quantized type is to use a
    quantization operation like `tf.quantization.quantize`. The inputs generated
    here are `tf.Tensor` objects that have been correctly quantized. This is the
    only way to provide a valid input that satisfies the operation's type
    constraints. The min/max scalar values are kept as numpy arrays as per
    the requested format.
    """
    list_of_inputs = []

    def create_quantized_input(name, input_float_np, bias_float_np, T_input, T_bias, out_type, min_in, max_in, min_b, max_b):
        input_float = tf.constant(input_float_np, dtype=tf.float32)
        bias_float = tf.constant(bias_float_np, dtype=tf.float32)
        
        input_quant, min_input_out, max_input_out = tf.quantization.quantize(
            input_float, min_in, max_in, T=T_input
        )
        bias_quant, min_bias_out, max_bias_out = tf.quantization.quantize(
            bias_float, min_b, max_b, T=T_bias
        )
        
        input_dict = {
            'input': input_quant,
            'bias': bias_quant,
            'min_input': np.array(min_input_out.numpy(), dtype=np.float32),
            'max_input': np.array(max_input_out.numpy(), dtype=np.float32),
            'min_bias': np.array(min_bias_out.numpy(), dtype=np.float32),
            'max_bias': np.array(max_bias_out.numpy(), dtype=np.float32),
            'out_type': out_type,
            'name': name
        }
        return input_dict

    list_of_inputs.append(create_quantized_input(
        'qint8_correct', np.array([[-10., 20.], [30., -40.]]), np.array([5., -5.]),
        tf.qint8, tf.qint8, tf.qint8, -50.0, 50.0, -10.0, 10.0
    ))
    
    list_of_inputs.append(create_quantized_input(
        'quint8_correct', np.array([[[10, 20], [40, 50]]], dtype=np.float32), np.array([5, 15], dtype=np.float32),
        tf.quint8, tf.quint8, tf.quint8, 0.0, 255.0, 0.0, 50.0
    ))

    list_of_inputs.append(create_quantized_input(
        'qint32_correct', np.array([[100000, -200000]], dtype=np.float32), np.array([50000], dtype=np.float32),
        tf.qint32, tf.qint32, tf.qint32, -300000.0, 300000.0, 0.0, 100000.0
    ))

    list_of_inputs.append(create_quantized_input(
        'qint16_correct', np.array([[1000, 2000], [-3000, -4000]], dtype=np.float32), np.array([500, -500], dtype=np.float32),
        tf.qint16, tf.qint16, tf.qint16, -5000.0, 5000.0, -1000.0, 1000.0
    ))

    list_of_inputs.append(create_quantized_input(
        'quint16_correct', np.array([[1000, 2000]], dtype=np.float32), np.array([100], dtype=np.float32),
        tf.quint16, tf.quint16, tf.quint16, 0.0, 65535.0, 0.0, 1000.0
    ))
    
    list_of_inputs.append(create_quantized_input(
        'mixed_types_correct', np.array([[10., 20.]], dtype=np.float32), np.array([5., -5.], dtype=np.float32),
        tf.quint8, tf.qint8, tf.qint32, 0.0, 25.5, -12.8, 12.7
    ))
    
    list_of_inputs.append(create_quantized_input(
        '4d_broadcast_correct', np.arange(24, dtype=np.float32).reshape(2, 2, 3, 2) - 12.0, np.array([10., -20.], dtype=np.float32),
        tf.qint8, tf.qint8, tf.qint8, -128.0, 127.0, -50.0, 50.0
    ))
    
    list_of_inputs.append(create_quantized_input(
        'upcast_correct', np.array([[120., 125.], [100., 110.]], dtype=np.float32), np.array([50., 60.], dtype=np.float32),
        tf.qint8, tf.qint8, tf.qint32, 0.0, 127.0, 0.0, 64.0
    ))

    list_of_inputs.append(create_quantized_input(
        'zero_range_correct', np.zeros((3, 5), dtype=np.float32), np.zeros(5, dtype=np.float32),
        tf.qint8, tf.qint8, tf.qint8, 0.0, 0.0, 0.0, 0.0
    ))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedBiasAdd"] = get_quantizedbiasadd_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedBiasAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedBiasAdd'.")

check_valid('tf.raw_ops.QuantizedBiasAdd', generated_inputs['tf.raw_ops.QuantizedBiasAdd'], lib="tf", suffix=0)
