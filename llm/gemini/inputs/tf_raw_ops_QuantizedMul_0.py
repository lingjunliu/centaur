
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def get_quantized_mul_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.QuantizedMul function.
    This is achieved by creating tf.Tensor objects with the required quantized dtypes
    (e.g., tf.qint8) using tf.quantization.quantize, as standard numpy types are not accepted.
    """
    list_of_inputs = []

    def create_quantized_tensors(x_float_val, y_float_val, min_x, max_x, min_y, max_y, T_x, T_y):
        """Helper to create valid quantized tensors for the API."""
        x_float = tf.constant(x_float_val, dtype=tf.float32)
        y_float = tf.constant(y_float_val, dtype=tf.float32)
        q_x, _, _ = tf.quantization.quantize(x_float, min_x, max_x, T=T_x, mode='SCALED')
        q_y, _, _ = tf.quantization.quantize(y_float, min_y, max_y, T=T_y, mode='SCALED')
        return q_x, q_y

    # Case 1: Basic qint8 x qint8 -> qint32
    q_x_1, q_y_1 = create_quantized_tensors([-1.0, 0.0, 1.0], [0.5, 1.0, 1.5], -2.0, 2.0, 0.0, 3.0, tf.qint8, tf.qint8)
    list_of_inputs.append({
        'x': q_x_1,
        'y': q_y_1,
        'min_x': tf.constant(-2.0, dtype=tf.float32),
        'max_x': tf.constant(2.0, dtype=tf.float32),
        'min_y': tf.constant(0.0, dtype=tf.float32),
        'max_y': tf.constant(3.0, dtype=tf.float32),
        'Toutput': tf.qint32,
        'name': 'valid_qint8_mul'
    })

    # Case 2: Basic quint8 x quint8 -> qint32
    q_x_2, q_y_2 = create_quantized_tensors([0.0, 12.8, 25.5], [1.0, 2.0, 3.0], 0.0, 25.5, 0.0, 30.0, tf.quint8, tf.quint8)
    list_of_inputs.append({
        'x': q_x_2,
        'y': q_y_2,
        'min_x': tf.constant(0.0, dtype=tf.float32),
        'max_x': tf.constant(25.5, dtype=tf.float32),
        'min_y': tf.constant(0.0, dtype=tf.float32),
        'max_y': tf.constant(30.0, dtype=tf.float32),
        'Toutput': tf.qint32,
        'name': 'valid_quint8_mul'
    })

    # Case 3: 2D tensors, qint16 x qint16 -> qint16
    q_x_3, q_y_3 = create_quantized_tensors([[-10.0, 20.0], [0.0, 300.0]], [[5.0, -2.0], [1.0, 1.0]], -500.0, 500.0, -10.0, 10.0, tf.qint16, tf.qint16)
    list_of_inputs.append({
        'x': q_x_3,
        'y': q_y_3,
        'min_x': tf.constant(-500.0, dtype=tf.float32),
        'max_x': tf.constant(500.0, dtype=tf.float32),
        'min_y': tf.constant(-10.0, dtype=tf.float32),
        'max_y': tf.constant(10.0, dtype=tf.float32),
        'Toutput': tf.qint16,
        'name': 'valid_2d_qint16'
    })

    # Case 4: quint16 x quint16 -> quint16
    q_x_4, q_y_4 = create_quantized_tensors([[0., 1000.], [2000., 60000.]], [[1., 2.], [0., 3.]], 0.0, 65535.0, 0.0, 4.0, tf.quint16, tf.quint16)
    list_of_inputs.append({
        'x': q_x_4,
        'y': q_y_4,
        'min_x': tf.constant(0.0, dtype=tf.float32),
        'max_x': tf.constant(65535.0, dtype=tf.float32),
        'min_y': tf.constant(0.0, dtype=tf.float32),
        'max_y': tf.constant(4.0, dtype=tf.float32),
        'Toutput': tf.quint16,
        'name': 'valid_quint16_mul'
    })

    # Case 5: qint32 x qint32 -> qint32
    q_x_5, q_y_5 = create_quantized_tensors([-100000.0, 200000.0], [10.0, -20.0], -210000.0, 210000.0, -30.0, 30.0, tf.qint32, tf.qint32)
    list_of_inputs.append({
        'x': q_x_5,
        'y': q_y_5,
        'min_x': tf.constant(-210000.0, dtype=tf.float32),
        'max_x': tf.constant(210000.0, dtype=tf.float32),
        'min_y': tf.constant(-30.0, dtype=tf.float32),
        'max_y': tf.constant(30.0, dtype=tf.float32),
        'Toutput': tf.qint32,
        'name': 'valid_qint32_mul'
    })

    # Case 6: Mismatched types qint8 x qint16 -> qint32
    q_x_6, q_y_6 = create_quantized_tensors([1.0, 2.0, -3.0], [100.0, -200.0, 300.0], -4.0, 4.0, -400.0, 400.0, tf.qint8, tf.qint16)
    list_of_inputs.append({
        'x': q_x_6,
        'y': q_y_6,
        'min_x': tf.constant(-4.0, dtype=tf.float32),
        'max_x': tf.constant(4.0, dtype=tf.float32),
        'min_y': tf.constant(-400.0, dtype=tf.float32),
        'max_y': tf.constant(400.0, dtype=tf.float32),
        'Toutput': tf.qint32,
        'name': 'valid_mismatched_qint8_qint16'
    })
    
    # Case 7: Mismatched types qint8 x quint8 -> qint8
    q_x_7, q_y_7 = create_quantized_tensors([-1.0, 0.5, 1.0], [0.1, 0.2, 0.3], -2.0, 2.0, 0.0, 1.0, tf.qint8, tf.quint8)
    list_of_inputs.append({
        'x': q_x_7,
        'y': q_y_7,
        'min_x': tf.constant(-2.0, dtype=tf.float32),
        'max_x': tf.constant(2.0, dtype=tf.float32),
        'min_y': tf.constant(0.0, dtype=tf.float32),
        'max_y': tf.constant(1.0, dtype=tf.float32),
        'Toutput': tf.qint8,
        'name': 'valid_mismatched_qint8_quint8'
    })

    # Case 8: 3D tensors qint8 x qint8 -> qint32
    x_float_8 = np.random.uniform(low=-10.0, high=10.0, size=(2, 1, 3)).astype(np.float32)
    y_float_8 = np.random.uniform(low=-10.0, high=10.0, size=(2, 1, 3)).astype(np.float32)
    q_x_8, q_y_8 = create_quantized_tensors(x_float_8, y_float_8, -12.0, 12.0, -12.0, 12.0, tf.qint8, tf.qint8)
    list_of_inputs.append({
        'x': q_x_8,
        'y': q_y_8,
        'min_x': tf.constant(-12.0, dtype=tf.float32),
        'max_x': tf.constant(12.0, dtype=tf.float32),
        'min_y': tf.constant(-12.0, dtype=tf.float32),
        'max_y': tf.constant(12.0, dtype=tf.float32),
        'Toutput': tf.qint32,
        'name': 'valid_3d_qint8'
    })

    # Case 9: Empty inputs
    q_x_9, q_y_9 = create_quantized_tensors([], [], -1.0, 1.0, -1.0, 1.0, tf.qint8, tf.qint8)
    list_of_inputs.append({
        'x': q_x_9,
        'y': q_y_9,
        'min_x': tf.constant(-1.0, dtype=tf.float32),
        'max_x': tf.constant(1.0, dtype=tf.float32),
        'min_y': tf.constant(-1.0, dtype=tf.float32),
        'max_y': tf.constant(1.0, dtype=tf.float32),
        'Toutput': tf.qint32,
        'name': 'valid_empty_inputs'
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedMul"] = get_quantized_mul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedMul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedMul'.")

check_valid('tf.raw_ops.QuantizedMul', generated_inputs['tf.raw_ops.QuantizedMul'], lib="tf", suffix=0)
