
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_quantization_fake_quant_with_min_max_args_gradient_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor, standard defaults
    gradients_1 = np.array([0.1, -0.2, 0.5, 1.2], dtype=np.float32)
    inputs_1 = np.array([-1.5, 2.0, 5.5, -7.0], dtype=np.float32)
    input_dict_1 = {
        "gradients": gradients_1,
        "inputs": inputs_1,
        "min": -6.0,
        "max": 6.0,
        "num_bits": 8,
        "narrow_range": False,
        "name": "fake_quant_grad_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D tensor, narrow range, 4 bits
    gradients_2 = np.ones((2, 3), dtype=np.float32) * 0.5
    inputs_2 = np.array([[-2.0, 0.0, 2.0], [-4.0, 1.0, 3.0]], dtype=np.float32)
    input_dict_2 = {
        "gradients": gradients_2,
        "inputs": inputs_2,
        "min": -3.0,
        "max": 3.0,
        "num_bits": 4,
        "narrow_range": True,
        "name": "fake_quant_grad_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D tensor, larger bits (16), asymmetric min/max
    gradients_3 = np.random.uniform(-1.0, 1.0, size=(2, 2, 2)).astype(np.float32)
    inputs_3 = np.random.uniform(-10.0, 10.0, size=(2, 2, 2)).astype(np.float32)
    input_dict_3 = {
        "gradients": gradients_3,
        "inputs": inputs_3,
        "min": -10.0,
        "max": 5.0,
        "num_bits": 16,
        "narrow_range": False,
        "name": "fake_quant_grad_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 4D tensor, narrow range True, small range
    gradients_4 = np.zeros((1, 2, 2, 1), dtype=np.float32)
    inputs_4 = np.array([[[[0.1], [0.2]], [[-0.1], [-0.2]]]], dtype=np.float32)
    input_dict_4 = {
        "gradients": gradients_4,
        "inputs": inputs_4,
        "min": -1.0,
        "max": 1.0,
        "num_bits": 8,
        "narrow_range": True,
        "name": "fake_quant_grad_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 1D tensor, low bit count (2 bits)
    gradients_5 = np.array([-0.5, 0.5], dtype=np.float32)
    inputs_5 = np.array([-2.5, 2.5], dtype=np.float32)
    input_dict_5 = {
        "gradients": gradients_5,
        "inputs": inputs_5,
        "min": -2.0,
        "max": 2.0,
        "num_bits": 2,
        "narrow_range": False,
        "name": "fake_quant_grad_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 2D tensor, positive values only
    gradients_6 = np.random.exponential(1.0, size=(3, 3)).astype(np.float32)
    inputs_6 = np.random.exponential(2.0, size=(3, 3)).astype(np.float32)
    input_dict_6 = {
        "gradients": gradients_6,
        "inputs": inputs_6,
        "min": 0.0,
        "max": 10.0,
        "num_bits": 8,
        "narrow_range": False,
        "name": "fake_quant_grad_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 3D tensor, negative values only
    gradients_7 = -np.random.exponential(1.0, size=(2, 3, 2)).astype(np.float32)
    inputs_7 = -np.random.exponential(2.0, size=(2, 3, 2)).astype(np.float32)
    input_dict_7 = {
        "gradients": gradients_7,
        "inputs": inputs_7,
        "min": -8.0,
        "max": 0.0,
        "num_bits": 8,
        "narrow_range": True,
        "name": "fake_quant_grad_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: High dimensional (5D) tensor
    gradients_8 = np.ones((1, 2, 2, 2, 1), dtype=np.float32) * -0.1
    inputs_8 = np.ones((1, 2, 2, 2, 1), dtype=np.float32) * 1.5
    input_dict_8 = {
        "gradients": gradients_8,
        "inputs": inputs_8,
        "min": -2.5,
        "max": 2.5,
        "num_bits": 12,
        "narrow_range": False,
        "name": "fake_quant_grad_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: 1D tensor, very narrow min/max range
    gradients_9 = np.array([10.0, -10.0, 5.0], dtype=np.float32)
    inputs_9 = np.array([0.001, -0.002, 0.005], dtype=np.float32)
    input_dict_9 = {
        "gradients": gradients_9,
        "inputs": inputs_9,
        "min": -0.01,
        "max": 0.01,
        "num_bits": 8,
        "narrow_range": False,
        "name": "fake_quant_grad_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 2D tensor, high quantization bit width (15)
    gradients_10 = np.random.normal(0.0, 1.0, size=(4, 4)).astype(np.float32)
    inputs_10 = np.random.normal(0.0, 5.0, size=(4, 4)).astype(np.float32)
    input_dict_10 = {
        "gradients": gradients_10,
        "inputs": inputs_10,
        "min": -15.0,
        "max": 15.0,
        "num_bits": 15,
        "narrow_range": True,
        "name": "fake_quant_grad_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.quantization.fake_quant_with_min_max_args_gradient"] = tf_quantization_fake_quant_with_min_max_args_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.quantization.fake_quant_with_min_max_args_gradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.quantization.fake_quant_with_min_max_args_gradient'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.quantization.fake_quant_with_min_max_args_gradient', generated_inputs['tf.quantization.fake_quant_with_min_max_args_gradient'], lib="tf", suffix=0)
