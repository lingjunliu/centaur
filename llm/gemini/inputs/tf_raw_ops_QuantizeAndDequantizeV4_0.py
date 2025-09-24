
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_quantize_and_dequantize_v4_inputs():
    list_of_inputs = []

    # Input 1: Basic float32, range determined from tensor
    input_val = np.array([[-1.0, 0.0, 1.0], [1.5, 0.5, -0.5]], dtype=np.float32)
    input_dict = {
        'input': input_val,
        'input_min': np.array(np.min(input_val), dtype=np.float32),
        'input_max': np.array(np.max(input_val), dtype=np.float32),
        'signed_input': True,
        'num_bits': 8,
        'range_given': False,
        'round_mode': 'HALF_TO_EVEN',
        'narrow_range': False,
        'axis': -1,
        'name': 'basic_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Range is given, float32
    input_val = np.array([-0.5, 0.1, 0.8, 1.2], dtype=np.float32)
    input_dict = {
        'input': input_val,
        'input_min': np.array(0.0, dtype=np.float32),
        'input_max': np.array(1.0, dtype=np.float32),
        'signed_input': True,
        'num_bits': 8,
        'range_given': True,
        'round_mode': 'HALF_TO_EVEN',
        'narrow_range': False,
        'axis': -1,
        'name': 'range_given_true'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Unsigned quantization
    input_val = np.array([0.0, 10.0, 50.0, 100.0, 255.0], dtype=np.float32)
    input_dict = {
        'input': input_val,
        'input_min': np.array(np.min(input_val), dtype=np.float32),
        'input_max': np.array(np.max(input_val), dtype=np.float32),
        'signed_input': False,
        'num_bits': 8,
        'range_given': False,
        'round_mode': 'HALF_TO_EVEN',
        'narrow_range': False,
        'axis': -1,
        'name': 'unsigned_quant'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different num_bits (4 bits)
    input_val = np.array([-8.0, -7.0, 0.0, 6.0, 7.0], dtype=np.float32)
    input_dict = {
        'input': input_val,
        'input_min': np.array(np.min(input_val), dtype=np.float32),
        'input_max': np.array(np.max(input_val), dtype=np.float32),
        'signed_input': True,
        'num_bits': 4,
        'range_given': False,
        'round_mode': 'HALF_TO_EVEN',
        'narrow_range': False,
        'axis': -1,
        'name': '4_bits_quant'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Narrow range
    input_val = np.array([-127.0, -64.0, 0.0, 64.0, 127.0], dtype=np.float32)
    input_dict = {
        'input': input_val,
        'input_min': np.array(np.min(input_val), dtype=np.float32),
        'input_max': np.array(np.max(input_val), dtype=np.float32),
        'signed_input': True,
        'num_bits': 8,
        'range_given': False,
        'round_mode': 'HALF_TO_EVEN',
        'narrow_range': True,
        'axis': -1,
        'name': 'narrow_range_true'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Round mode 'HALF_UP'
    input_val = np.array([-7.5, -2.5, 2.5, 7.5], dtype=np.float32)
    input_dict = {
        'input': input_val,
        'input_min': np.array(-8.0, dtype=np.float32),
        'input_max': np.array(8.0, dtype=np.float32),
        'signed_input': True,
        'num_bits': 8,
        'range_given': True,
        'round_mode': 'HALF_UP',
        'narrow_range': False,
        'axis': -1,
        'name': 'round_half_up'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: float64 dtype
    input_val = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    input_dict = {
        'input': input_val,
        'input_min': np.array(np.min(input_val), dtype=np.float64),
        'input_max': np.array(np.max(input_val), dtype=np.float64),
        'signed_input': True,
        'num_bits': 8,
        'range_given': False,
        'round_mode': 'HALF_TO_EVEN',
        'narrow_range': False,
        'axis': -1,
        'name': 'float64_dtype'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: float64 with range_given=True
    input_val = np.random.uniform(-200, 200, size=(5,)).astype(np.float64)
    input_dict = {
        'input': input_val,
        'input_min': np.array(-150.0, dtype=np.float64),
        'input_max': np.array(150.0, dtype=np.float64),
        'signed_input': True,
        'num_bits': 10,
        'range_given': True,
        'round_mode': 'HALF_TO_EVEN',
        'narrow_range': False,
        'axis': -1,
        'name': 'float64_range_given'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D input tensor
    input_val = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    input_dict = {
        'input': input_val,
        'input_min': np.array(0.0, dtype=np.float32),
        'input_max': np.array(10.0, dtype=np.float32),
        'signed_input': False,
        'num_bits': 8,
        'range_given': True,
        'round_mode': 'HALF_UP',
        'narrow_range': True,
        'axis': -1,
        'name': '1d_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: All input values are the same
    input_val = np.full((2, 2), 5.5, dtype=np.float32)
    input_dict = {
        'input': input_val,
        'input_min': np.array(5.5, dtype=np.float32),
        'input_max': np.array(5.5, dtype=np.float32),
        'signed_input': True,
        'num_bits': 8,
        'range_given': False,
        'round_mode': 'HALF_TO_EVEN',
        'narrow_range': False,
        'axis': -1,
        'name': 'all_same_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: 4D input tensor
    input_val = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        'input': input_val,
        'input_min': np.array(np.min(input_val), dtype=np.float32),
        'input_max': np.array(np.max(input_val), dtype=np.float32),
        'signed_input': True,
        'num_bits': 16,
        'range_given': False,
        'round_mode': 'HALF_TO_EVEN',
        'narrow_range': False,
        'axis': -1,
        'name': '4d_input_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizeAndDequantizeV4"] = tf_raw_ops_quantize_and_dequantize_v4_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizeAndDequantizeV4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizeAndDequantizeV4'.")

check_valid('tf.raw_ops.QuantizeAndDequantizeV4', generated_inputs['tf.raw_ops.QuantizeAndDequantizeV4'], lib="tf", suffix=0)
