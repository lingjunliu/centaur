
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_quantization_fake_quant_with_min_max_vars_gradient_inputs():
    list_of_inputs = []

    # Input 1
    gradients = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    inputs = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(3.0, dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test_op_1"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    gradients = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    inputs = np.array([-0.5, -1.5, -2.5], dtype=np.float32)
    min_val = np.array(-3.0, dtype=np.float32)
    max_val = np.array(0.0, dtype=np.float32)
    num_bits = 4
    narrow_range = True
    name = "test_op_2"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    gradients = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    inputs = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(4.0, dtype=np.float32)
    num_bits = 6
    narrow_range = False
    name = "test_op_3"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    gradients = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    inputs = np.array([[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]], dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(8.0, dtype=np.float32)
    num_bits = 2
    narrow_range = True
    name = "test_op_4"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    gradients = np.array([1.0], dtype=np.float32)
    inputs = np.array([0.5], dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(1.0, dtype=np.float32)
    num_bits = 7
    narrow_range = False
    name = "test_op_5"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    gradients = np.array([-5.0], dtype=np.float32)
    inputs = np.array([-2.5], dtype=np.float32)
    min_val = np.array(-5.0, dtype=np.float32)
    max_val = np.array(-1.0, dtype=np.float32)
    num_bits = 3
    narrow_range = True
    name = "test_op_6"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    gradients = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    inputs = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(1.0, dtype=np.float32)
    num_bits = 8
    narrow_range = True
    name = "test_op_7"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    gradients = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    inputs = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    min_val = np.array(-2.0, dtype=np.float32)
    max_val = np.array(3.0, dtype=np.float32)
    num_bits = 5
    narrow_range = False
    name = "test_op_8"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    gradients = np.array(1.0, dtype=np.float32)
    inputs = np.array(0.5, dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(1.0, dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test_op_9"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    gradients = np.array(-1.0, dtype=np.float32)
    inputs = np.array(-0.5, dtype=np.float32)
    min_val = np.array(-1.0, dtype=np.float32)
    max_val = np.array(0.0, dtype=np.float32)
    num_bits = 6
    narrow_range = True
    name = "test_op_10"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.quantization.fake_quant_with_min_max_vars_gradient"] = tf_quantization_fake_quant_with_min_max_vars_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.quantization.fake_quant_with_min_max_vars_gradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.quantization.fake_quant_with_min_max_vars_gradient'.")

check_valid('tf.quantization.fake_quant_with_min_max_vars_gradient', generated_inputs['tf.quantization.fake_quant_with_min_max_vars_gradient'], lib="tf", suffix=0)
