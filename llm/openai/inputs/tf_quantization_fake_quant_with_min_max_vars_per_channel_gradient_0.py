
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_quantization_fake_quant_with_min_max_vars_per_channel_gradient_inputs():
    list_of_inputs = []

    # Input 1: 1D
    gradients = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    inputs = np.array([-0.2, 0.0, 1.1], dtype=np.float32)
    min_arr = np.array([-1.0, -0.5, 0.0], dtype=np.float32)
    max_arr = np.array([1.0, 0.5, 2.0], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 8,
        "narrow_range": False,
        "name": "case1_1d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D (b=2, d=4)
    gradients = np.array([[0.1, -0.2, 0.3, -0.4],
                          [1.0, -1.0, 0.5, -0.5]], dtype=np.float32)
    inputs = np.array([[0.05, -0.25, 0.35, -0.45],
                       [0.9, -1.2, 0.6, -0.55]], dtype=np.float32)
    min_arr = np.array([-0.5, -0.3, -0.2, -0.1], dtype=np.float32)
    max_arr = np.array([0.5, 0.7, 0.8, 0.9], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 7,
        "narrow_range": False,
        "name": "case2_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D (b=1, h=2, w=2, d=3)
    gradients = np.array([[[[0.1, -0.2, 0.3],
                            [0.2, -0.1, 0.0]],
                           [[-0.3, 0.4, -0.5],
                            [0.6, -0.7, 0.8]]]], dtype=np.float32)
    inputs = np.array([[[[0.05, -0.1, 0.25],
                         [0.15, -0.05, 0.05]],
                        [[-0.25, 0.35, -0.45],
                         [0.55, -0.65, 0.75]]]], dtype=np.float32)
    min_arr = np.array([-0.6, -0.4, -0.2], dtype=np.float32)
    max_arr = np.array([0.6, 0.8, 0.9], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 8,
        "narrow_range": True,
        "name": "case3_4d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D (b=4, h=3, w=3, d=1)
    gradients = np.random.uniform(-1, 1, size=(4, 3, 3, 1)).astype(np.float32)
    inputs = np.random.uniform(-2, 2, size=(4, 3, 3, 1)).astype(np.float32)
    min_arr = np.array([-1.0], dtype=np.float32)
    max_arr = np.array([1.0], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 9,
        "narrow_range": False,
        "name": "case4_4d_d1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D (b=5, d=2) narrow_range True
    gradients = np.array([[0.2, -0.2],
                          [0.3, -0.1],
                          [-0.4, 0.4],
                          [0.0, 0.0],
                          [1.0, -1.0]], dtype=np.float32)
    inputs = np.array([[0.25, -0.25],
                       [0.35, -0.15],
                       [-0.45, 0.45],
                       [0.05, -0.05],
                       [1.2, -1.2]], dtype=np.float32)
    min_arr = np.array([-0.3, -0.6], dtype=np.float32)
    max_arr = np.array([0.7, 0.9], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 8,
        "narrow_range": True,
        "name": "case5_2d_narrow"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D (d=1)
    gradients = np.array([2.5], dtype=np.float32)
    inputs = np.array([-10.0], dtype=np.float32)
    min_arr = np.array([-6.0], dtype=np.float32)
    max_arr = np.array([6.0], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 12,
        "narrow_range": False,
        "name": "case6_1d_single"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D (b=2, h=1, w=5, d=5)
    gradients = np.linspace(-0.5, 0.5, num=2*1*5*5).reshape(2, 1, 5, 5).astype(np.float32)
    inputs = np.linspace(-1.0, 1.0, num=2*1*5*5).reshape(2, 1, 5, 5).astype(np.float32)
    min_arr = np.array([-0.9, -0.5, -0.1, 0.0, 0.2], dtype=np.float32)
    max_arr = np.array([0.9, 0.6, 0.4, 0.7, 1.0], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 6,
        "narrow_range": False,
        "name": "case7_4d_wide_d5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D (b=1, d=8), num_bits=16
    gradients = np.array([[0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4]], dtype=np.float32)
    inputs = np.array([[0.05, -0.15, 0.25, -0.35, 0.45, -0.55, 0.65, -0.75]], dtype=np.float32)
    min_arr = np.array([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2], dtype=np.float32)
    max_arr = np.array([0.8, 0.7, 0.9, 1.0, 1.2, 0.6, 0.5, 0.4], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 16,
        "narrow_range": False,
        "name": "case8_2d_16bits"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D (d=7), fractional min/max
    gradients = np.array([-0.05, 0.1, -0.15, 0.2, -0.25, 0.3, -0.35], dtype=np.float32)
    inputs = np.array([0.01, -0.12, 0.23, -0.34, 0.45, -0.56, 0.67], dtype=np.float32)
    min_arr = np.array([-0.25, -0.2, -0.15, -0.1, -0.05, -0.02, -0.01], dtype=np.float32)
    max_arr = np.array([0.3, 0.25, 0.35, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 10,
        "narrow_range": True,
        "name": "case9_1d_frac"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D (b=3, h=2, w=2, d=2), num_bits=2
    gradients = np.random.randn(3, 2, 2, 2).astype(np.float32)
    inputs = np.random.uniform(-0.8, 0.8, size=(3, 2, 2, 2)).astype(np.float32)
    min_arr = np.array([-0.7, -0.4], dtype=np.float32)
    max_arr = np.array([0.7, 0.9], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 2,
        "narrow_range": True,
        "name": "case10_4d_lowbits"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D (b=3, d=3), negative ranges
    gradients = np.array([[1.0, -1.0, 0.5],
                          [-0.3, 0.2, -0.1],
                          [0.0, 0.0, 0.0]], dtype=np.float32)
    inputs = np.array([[-1.5, -0.8, -0.2],
                       [-0.6, -0.4, -0.05],
                       [-0.9, -0.1, -0.3]], dtype=np.float32)
    min_arr = np.array([-2.0, -1.0, -0.5], dtype=np.float32)
    max_arr = np.array([-0.5, -0.2, -0.1], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 5,
        "narrow_range": False,
        "name": "case11_2d_negative_range"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 4D (b=1, h=4, w=1, d=4), mixed values
    gradients = np.array([[[[0.05, -0.05, 0.1, -0.1]],
                           [[0.2, -0.2, 0.3, -0.3]],
                           [[-0.4, 0.4, -0.5, 0.5]],
                           [[0.6, -0.6, 0.7, -0.7]]]], dtype=np.float32)
    inputs = np.array([[[[0.01, -0.02, 0.03, -0.04]],
                        [[0.25, -0.15, 0.35, -0.45]],
                        [[-0.55, 0.65, -0.75, 0.85]],
                        [[0.95, -1.05, 1.15, -1.25]]]], dtype=np.float32)
    min_arr = np.array([-0.3, -0.2, -0.1, -0.05], dtype=np.float32)
    max_arr = np.array([0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 11,
        "narrow_range": False,
        "name": "case12_4d_varied"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient"] = tf_quantization_fake_quant_with_min_max_vars_per_channel_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient', generated_inputs['tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient'], lib="tf", suffix=0)
