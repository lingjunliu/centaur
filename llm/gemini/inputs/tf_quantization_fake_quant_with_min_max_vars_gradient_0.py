
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_quantization_fake_quant_with_min_max_vars_gradient_inputs():
    list_of_inputs = []

    # Case 1: 1D input, 8 bits, standard range
    input_dict = {
        "gradients": np.array([0.1, -0.2, 0.3], dtype=np.float32),
        "inputs": np.array([0.5, -0.5, 1.2], dtype=np.float32),
        "min": np.array(-1.0, dtype=np.float32),
        "max": np.array(1.0, dtype=np.float32),
        "num_bits": 8,
        "narrow_range": False,
        "name": "case1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D input, 4 bits, narrow range
    input_dict = {
        "gradients": np.array([[0.5, -0.1], [0.2, -0.3]], dtype=np.float32),
        "inputs": np.array([[1.5, -1.5], [0.5, -0.5]], dtype=np.float32),
        "min": np.array(-2.0, dtype=np.float32),
        "max": np.array(2.0, dtype=np.float32),
        "num_bits": 4,
        "narrow_range": True,
        "name": "case2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 3D input, 2 bits, standard range
    input_dict = {
        "gradients": np.ones((2, 2, 2), dtype=np.float32) * 0.1,
        "inputs": np.zeros((2, 2, 2), dtype=np.float32),
        "min": np.array(-0.5, dtype=np.float32),
        "max": np.array(0.5, dtype=np.float32),
        "num_bits": 2,
        "narrow_range": False,
        "name": "case3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 4D input, 7 bits, narrow range
    input_dict = {
        "gradients": np.random.randn(1, 2, 2, 3).astype(np.float32),
        "inputs": np.random.randn(1, 2, 2, 3).astype(np.float32),
        "min": np.array(-3.0, dtype=np.float32),
        "max": np.array(3.0, dtype=np.float32),
        "num_bits": 7,
        "narrow_range": True,
        "name": "case4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 1D size 1 input, 8 bits
    input_dict = {
        "gradients": np.array([1.5], dtype=np.float32),
        "inputs": np.array([0.0], dtype=np.float32),
        "min": np.array(-1.0, dtype=np.float32),
        "max": np.array(1.0, dtype=np.float32),
        "num_bits": 8,
        "narrow_range": False,
        "name": "case5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Large values, 5 bits, standard range
    input_dict = {
        "gradients": np.array([[10., 20.], [-10., -20.]], dtype=np.float32),
        "inputs": np.array([[5., 15.], [-5., -15.]], dtype=np.float32),
        "min": np.array(-10.0, dtype=np.float32),
        "max": np.array(10.0, dtype=np.float32),
        "num_bits": 5,
        "narrow_range": False,
        "name": "case6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Positive min and max, 8 bits
    input_dict = {
        "gradients": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "inputs": np.array([[1.5, 2.5], [2.0, 3.5]], dtype=np.float32),
        "min": np.array(1.0, dtype=np.float32),
        "max": np.array(3.0, dtype=np.float32),
        "num_bits": 8,
        "narrow_range": False,
        "name": "case7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Negative min and max, 6 bits, narrow range
    input_dict = {
        "gradients": np.array([[-0.1, -0.2], [-0.3, -0.4]], dtype=np.float32),
        "inputs": np.array([[-1.5, -2.5], [-2.0, -3.5]], dtype=np.float32),
        "min": np.array(-5.0, dtype=np.float32),
        "max": np.array(-1.0, dtype=np.float32),
        "num_bits": 6,
        "narrow_range": True,
        "name": "case8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: Very small range, 3 bits
    input_dict = {
        "gradients": np.array([0.01, -0.02, 0.03, -0.04, 0.05], dtype=np.float32),
        "inputs": np.array([0.005, -0.005, 0.012, -0.015, 0.02], dtype=np.float32),
        "min": np.array(-0.01, dtype=np.float32),
        "max": np.array(0.01, dtype=np.float32),
        "num_bits": 3,
        "narrow_range": False,
        "name": "case9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: 1D array of size 10, zero gradients, 8 bits
    input_dict = {
        "gradients": np.zeros((10,), dtype=np.float32),
        "inputs": np.ones((10,), dtype=np.float32),
        "min": np.array(0.0, dtype=np.float32),
        "max": np.array(1.0, dtype=np.float32),
        "num_bits": 8,
        "narrow_range": True,
        "name": "case10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.quantization.fake_quant_with_min_max_vars_gradient"] = tf_quantization_fake_quant_with_min_max_vars_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.quantization.fake_quant_with_min_max_vars_gradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.quantization.fake_quant_with_min_max_vars_gradient'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.quantization.fake_quant_with_min_max_vars_gradient', generated_inputs['tf.quantization.fake_quant_with_min_max_vars_gradient'], lib="tf", suffix=0)
