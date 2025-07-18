
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_quantization_dequantize_inputs():
    list_of_inputs = []

    # Helper function to generate quantized values as numpy arrays.
    # This adheres to the requirement of providing inputs in numpy format,
    # which resolves the ValueError from the test harness.
    def create_quantized_input_numpy(input_float, min_val, max_val, q_type, mode, narrow_range=False, axis=None):
        q_input_tensor, q_min_tensor, q_max_tensor = tf.quantization.quantize(
            tf.constant(input_float, dtype=tf.float32), 
            tf.constant(min_val, dtype=tf.float32), 
            tf.constant(max_val, dtype=tf.float32),
            T=q_type, 
            mode=mode, 
            narrow_range=narrow_range, 
            axis=axis
        )
        # Convert all tensors to numpy arrays, which loses the special quantized dtype information.
        return q_input_tensor.numpy(), q_min_tensor.numpy(), q_max_tensor.numpy()

    # Input 1: Basic quint8, MIN_COMBINED, per-tensor
    q_input, q_min, q_max = create_quantized_input_numpy(
        np.array([[0, 1, 2], [3, 4, 6]], dtype=np.float32), 0.0, 6.0, tf.quint8, 'MIN_COMBINED'
    )
    input_dict = {
        'input': q_input,
        'min_range': q_min,
        'max_range': q_max,
        'mode': 'MIN_COMBINED',
        'narrow_range': False,
        'axis': -1,
        'dtype': tf.float32,
        'name': 'test_quint8_min_combined'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic qint8, MIN_COMBINED, per-tensor
    q_input, q_min, q_max = create_quantized_input_numpy(
        np.array([[-10, 0], [10, 20]], dtype=np.float32), -20.0, 20.0, tf.qint8, 'MIN_COMBINED'
    )
    input_dict = {
        'input': q_input,
        'min_range': q_min,
        'max_range': q_max,
        'mode': 'MIN_COMBINED',
        'narrow_range': False,
        'axis': -1,
        'dtype': tf.float32,
        'name': 'test_qint8_min_combined'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: MIN_FIRST mode, quint8, 1D tensor
    q_input, q_min, q_max = create_quantized_input_numpy(
        np.array([0, 1, 2.5, 5.0, 10.0], dtype=np.float32), 0.0, 10.0, tf.quint8, 'MIN_FIRST'
    )
    input_dict = {
        'input': q_input,
        'min_range': q_min,
        'max_range': q_max,
        'mode': 'MIN_FIRST',
        'narrow_range': False,
        'axis': -1,
        'dtype': tf.float32,
        'name': 'test_min_first'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: SCALED mode, qint8, per-tensor
    q_input, q_min, q_max = create_quantized_input_numpy(
        np.array([[-1, -0.5], [0, 0.5], [1, 2]], dtype=np.float32), -1.0, 1.0, tf.qint8, 'SCALED'
    )
    input_dict = {
        'input': q_input,
        'min_range': q_min,
        'max_range': q_max,
        'mode': 'SCALED',
        'narrow_range': False,
        'axis': -1,
        'dtype': tf.float32,
        'name': 'test_scaled'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Per-channel quantization, MIN_COMBINED, axis=1
    q_input, q_min, q_max = create_quantized_input_numpy(
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        np.array([0.0, 2.0, 3.0], dtype=np.float32),
        np.array([4.0, 5.0, 7.0], dtype=np.float32),
        tf.quint8, 'MIN_COMBINED', axis=1
    )
    input_dict = {
        'input': q_input,
        'min_range': q_min,
        'max_range': q_max,
        'mode': 'MIN_COMBINED',
        'narrow_range': False,
        'axis': 1,
        'dtype': tf.float32,
        'name': 'test_per_channel'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Output dtype=bfloat16
    q_input, q_min, q_max = create_quantized_input_numpy(
        np.array([0, 1, 2, 3, 4, 5], dtype=np.float32), 0.0, 5.0, tf.quint8, 'MIN_COMBINED'
    )
    input_dict = {
        'input': q_input,
        'min_range': q_min,
        'max_range': q_max,
        'mode': 'MIN_COMBINED',
        'narrow_range': False,
        'axis': -1,
        'dtype': tf.bfloat16,
        'name': 'test_bfloat16_output'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: qint16 input type
    q_input, q_min, q_max = create_quantized_input_numpy(
        np.linspace(-1000, 1000, 10, dtype=np.float32), -1000.0, 1000.0, tf.qint16, 'MIN_COMBINED'
    )
    input_dict = {
        'input': q_input,
        'min_range': q_min,
        'max_range': q_max,
        'mode': 'MIN_COMBINED',
        'narrow_range': False,
        'axis': -1,
        'dtype': tf.float32,
        'name': 'test_qint16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: quint16 input type
    q_input, q_min, q_max = create_quantized_input_numpy(
        np.linspace(0, 50000, 20, dtype=np.float32), 0.0, 50000.0, tf.quint16, 'MIN_COMBINED'
    )
    input_dict = {
        'input': q_input,
        'min_range': q_min,
        'max_range': q_max,
        'mode': 'MIN_COMBINED',
        'narrow_range': False,
        'axis': -1,
        'dtype': tf.float32,
        'name': 'test_quint16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.quantization.dequantize"] = tf_quantization_dequantize_inputs()

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
