
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_dequantize_inputs():
    list_of_inputs = []

    # Case 1: qint8, MIN_COMBINED
    min_range_1 = -10.0
    max_range_1 = 10.0
    input_tensor_1 = tf.quantization.quantize(
        tf.constant([-10., 0., 5., 10.], dtype=tf.float32),
        min_range=min_range_1, max_range=max_range_1, T=tf.qint8, mode='MIN_COMBINED')[0].numpy()
    input_dict = {
        'input': input_tensor_1,
        'min_range': np.array(min_range_1, dtype=np.float32),
        'max_range': np.array(max_range_1, dtype=np.float32),
        'mode': 'MIN_COMBINED',
        'narrow_range': False,
        'axis': -1,
        'dtype': tf.float32,
        'name': 'test_qint8_min_combined'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: qint8, MIN_FIRST
    min_range_2 = -128.0
    max_range_2 = 127.0
    input_tensor_2 = tf.quantization.quantize(
        tf.constant([-128, -64, 0, 64, 127], dtype=tf.float32),
        min_range=min_range_2, max_range=max_range_2, T=tf.qint8, mode='MIN_FIRST')[0].numpy()
    input_dict = {
        'input': input_tensor_2,
        'min_range': np.array(min_range_2, dtype=np.float32),
        'max_range': np.array(max_range_2, dtype=np.float32),
        'mode': 'MIN_FIRST',
        'narrow_range': False,
        'axis': -1,
        'dtype': tf.float32,
        'name': 'test_qint8_min_first'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: qint8, SCALED
    min_range_3 = -1.0
    max_range_3 = 1.0
    input_tensor_3 = tf.quantization.quantize(
        tf.constant([-1., -0.5, 0., 0.5, 1.], dtype=tf.float32),
        min_range=min_range_3, max_range=max_range_3, T=tf.qint8, mode='SCALED')[0].numpy()
    input_dict = {
        'input': input_tensor_3,
        'min_range': np.array(min_range_3, dtype=np.float32),
        'max_range': np.array(max_range_3, dtype=np.float32),
        'mode': 'SCALED',
        'narrow_range': False,
        'axis': -1,
        'dtype': tf.float32,
        'name': 'test_qint8_scaled'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: qint8, SCALED, narrow_range=True
    min_range_4 = -1.0
    max_range_4 = 1.0
    input_tensor_4 = tf.quantization.quantize(
        tf.constant([-1., -0.5, 0., 0.5, 0.9], dtype=tf.float32),
        min_range=min_range_4, max_range=max_range_4, T=tf.qint8, mode='SCALED', narrow_range=True)[0].numpy()
    input_dict = {
        'input': input_tensor_4,
        'min_range': np.array(min_range_4, dtype=np.float32),
        'max_range': np.array(max_range_4, dtype=np.float32),
        'mode': 'SCALED',
        'narrow_range': True,
        'axis': -1,
        'dtype': tf.float32,
        'name': 'test_qint8_scaled_narrow'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: qint16, MIN_COMBINED
    min_range_5 = -30000.0
    max_range_5 = 30000.0
    input_tensor_5 = tf.quantization.quantize(
        tf.constant([-25000., 0., 28000.], dtype=tf.float32),
        min_range=min_range_5, max_range=max_range_5, T=tf.qint16, mode='MIN_COMBINED')[0].numpy()
    input_dict = {
        'input': input_tensor_5,
        'min_range': np.array(min_range_5, dtype=np.float32),
        'max_range': np.array(max_range_5, dtype=np.float32),
        'mode': 'MIN_COMBINED',
        'narrow_range': False,
        'axis': -1,
        'dtype': tf.float32,
        'name': 'test_qint16_min_combined'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: qint32, SCALED
    min_range_6 = -200000.0
    max_range_6 = 200000.0
    input_tensor_6 = tf.quantization.quantize(
        tf.constant([-150000, 0, 180000], dtype=tf.float32),
        min_range=min_range_6, max_range=max_range_6, T=tf.qint32, mode='SCALED')[0].numpy()
    input_dict = {
        'input': input_tensor_6,
        'min_range': np.array(min_range_6, dtype=np.float32),
        'max_range': np.array(max_range_6, dtype=np.float32),
        'mode': 'SCALED',
        'narrow_range': False,
        'axis': -1,
        'dtype': tf.float32,
        'name': 'test_qint32_scaled'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Per-channel qint8, MIN_COMBINED, axis=1
    min_range_7 = np.array([-10., -20.], dtype=np.float32)
    max_range_7 = np.array([10., 20.], dtype=np.float32)
    input_tensor_7 = tf.quantization.quantize(
        tf.constant([[-9., 8., -1.], [-18., 15., 0.]], dtype=tf.float32),
        min_range=min_range_7, max_range=max_range_7, T=tf.qint8, axis=1)[0].numpy()
    input_dict = {
        'input': input_tensor_7,
        'min_range': min_range_7,
        'max_range': max_range_7,
        'mode': 'MIN_COMBINED',
        'narrow_range': False,
        'axis': 1,
        'dtype': tf.float32,
        'name': 'test_per_channel_qint8_min_combined'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Per-channel qint8, SCALED, axis=0
    min_range_8 = np.array([-1., -2., -3.], dtype=np.float32)
    max_range_8 = np.array([1., 2., 3.], dtype=np.float32)
    input_tensor_8 = tf.quantization.quantize(
        tf.constant([[-0.5, 0.5], [-1.5, 1.5], [-2.5, 2.5]], dtype=tf.float32),
        min_range=min_range_8, max_range=max_range_8, T=tf.qint8, mode='SCALED', axis=0)[0].numpy()
    input_dict = {
        'input': input_tensor_8,
        'min_range': min_range_8,
        'max_range': max_range_8,
        'mode': 'SCALED',
        'narrow_range': False,
        'axis': 0,
        'dtype': tf.float32,
        'name': 'test_per_channel_qint8_scaled'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: qint8, bfloat16 output
    min_range_9 = -10.0
    max_range_9 = 10.0
    input_tensor_9 = tf.quantization.quantize(
        tf.constant([-8., 0., 8.], dtype=tf.float32),
        min_range=min_range_9, max_range=max_range_9, T=tf.qint8, mode='MIN_COMBINED')[0].numpy()
    input_dict = {
        'input': input_tensor_9,
        'min_range': np.array(min_range_9, dtype=np.float32),
        'max_range': np.array(max_range_9, dtype=np.float32),
        'mode': 'MIN_COMBINED',
        'narrow_range': False,
        'axis': -1,
        'dtype': tf.bfloat16,
        'name': 'test_qint8_bfloat16_output'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: Per-channel qint16, MIN_FIRST, axis=-1
    min_range_10 = np.array([-1000, -2000, -3000], dtype=np.float32)
    max_range_10 = np.array([1000, 2000, 3000], dtype=np.float32)
    input_tensor_10 = tf.quantization.quantize(
        tf.constant([[-500, -1500, -2500], [500, 1500, 2500]], dtype=tf.float32),
        min_range=min_range_10, max_range=max_range_10, T=tf.qint16, mode='MIN_FIRST', axis=-1)[0].numpy()
    input_dict = {
        'input': input_tensor_10,
        'min_range': min_range_10,
        'max_range': max_range_10,
        'mode': 'MIN_FIRST',
        'narrow_range': False,
        'axis': -1,
        'dtype': tf.float32,
        'name': 'test_per_channel_qint16_min_first'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 11: 3D input tensor, qint8
    min_range_11 = -5.0
    max_range_11 = 5.0
    input_tensor_11 = tf.quantization.quantize(
        tf.random.uniform(shape=(2,3,4), minval=-5, maxval=5, dtype=tf.float32),
        min_range=min_range_11, max_range=max_range_11, T=tf.qint8, mode='SCALED')[0].numpy()
    input_dict = {
        'input': input_tensor_11,
        'min_range': np.array(min_range_11, dtype=np.float32),
        'max_range': np.array(max_range_11, dtype=np.float32),
        'mode': 'SCALED',
        'narrow_range': False,
        'axis': -1,
        'dtype': tf.float32,
        'name': 'test_qint8_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
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
