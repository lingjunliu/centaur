
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_quantization_fake_quant_with_min_max_args_inputs():
    list_of_inputs = []
    
    # Input 1
    list_of_inputs.append({
        'inputs': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        'min': -5.0,
        'max': 5.0,
        'num_bits': 8,
        'narrow_range': False,
        'name': "quant_1"
    })
    
    # Input 2
    list_of_inputs.append({
        'inputs': np.array([[-10.0, -5.0], [0.0, 5.0]], dtype=np.float32),
        'min': -10.0,
        'max': 10.0,
        'num_bits': 16,
        'narrow_range': True,
        'name': "quant_2"
    })
    
    # Input 3
    list_of_inputs.append({
        'inputs': np.array([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]], dtype=np.float32),
        'min': 0.0,
        'max': 9.0,
        'num_bits': 4,
        'narrow_range': False,
        'name': "quant_3"
    })
    
    # Input 4
    list_of_inputs.append({
        'inputs': np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        'min': -2.0,
        'max': 2.0,
        'num_bits': 2,
        'narrow_range': True,
        'name': "quant_4"
    })
    
    # Input 5
    list_of_inputs.append({
        'inputs': np.array([[-15.0, -12.0, -9.0], [9.0, 12.0, 15.0]], dtype=np.float32),
        'min': -12.0,
        'max': 12.0,
        'num_bits': 12,
        'narrow_range': False,
        'name': "quant_5"
    })
    
    # Input 6
    list_of_inputs.append({
        'inputs': np.array([-0.5, 0.5], dtype=np.float32),
        'min': -1.0,
        'max': 1.0,
        'num_bits': 8,
        'narrow_range': True,
        'name': "quant_6"
    })
    
    # Input 7
    list_of_inputs.append({
        'inputs': np.array([[-45.0, -25.0], [25.0, 45.0]], dtype=np.float32),
        'min': -50.0,
        'max': 50.0,
        'num_bits': 15,
        'narrow_range': False,
        'name': "quant_7"
    })
    
    # Input 8
    list_of_inputs.append({
        'inputs': np.array([0.1, 0.2, 0.3], dtype=np.float32),
        'min': 0.0,
        'max': 1.0,
        'num_bits': 8,
        'narrow_range': True,
        'name': "quant_8"
    })
    
    # Input 9
    list_of_inputs.append({
        'inputs': np.array([-10.0, -8.0, -6.0], dtype=np.float32),
        'min': -12.0,
        'max': -2.0,
        'num_bits': 10,
        'narrow_range': False,
        'name': "quant_9"
    })
    
    # Input 10
    list_of_inputs.append({
        'inputs': np.zeros((3, 3), dtype=np.float32),
        'min': -6.0,
        'max': 6.0,
        'num_bits': 8,
        'narrow_range': False,
        'name': "quant_10"
    })
    
    return list_of_inputs

generated_inputs["tf.quantization.fake_quant_with_min_max_args"] = tf_quantization_fake_quant_with_min_max_args_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.quantization.fake_quant_with_min_max_args' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.quantization.fake_quant_with_min_max_args'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.quantization.fake_quant_with_min_max_args', generated_inputs['tf.quantization.fake_quant_with_min_max_args'], lib="tf", suffix=0)
