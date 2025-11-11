
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_quantization_fake_quant_with_min_max_vars_per_channel_gradient_inputs():
    list_of_inputs = []

    gradients = np.random.rand(4, 3, 2, 5).astype(np.float32)
    inputs = np.random.rand(4, 3, 2, 5).astype(np.float32)
    min_val = np.random.rand(5).astype(np.float32)
    max_val = np.random.rand(5).astype(np.float32)
    num_bits = 8
    narrow_range = False
    name = "fake_quant_test_1"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    gradients = np.random.rand(2, 7).astype(np.float32)
    inputs = np.random.rand(2, 7).astype(np.float32)
    min_val = np.random.rand(7).astype(np.float32)
    max_val = np.random.rand(7).astype(np.float32)
    num_bits = 4
    narrow_range = True
    name = "fake_quant_test_2"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    gradients = np.random.rand(3).astype(np.float32)
    inputs = np.random.rand(3).astype(np.float32)
    min_val = np.random.rand(3).astype(np.float32)
    max_val = np.random.rand(3).astype(np.float32)
    num_bits = 12
    narrow_range = False
    name = "fake_quant_test_3"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    gradients = np.random.rand(5, 4).astype(np.float32)
    inputs = np.random.rand(5, 4).astype(np.float32)
    min_val = np.random.rand(4).astype(np.float32)
    max_val = np.random.rand(4).astype(np.float32)
    num_bits = 2
    narrow_range = True
    name = "fake_quant_test_4"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    gradients = np.random.rand(10, 1, 1, 1).astype(np.float32)
    inputs = np.random.rand(10, 1, 1, 1).astype(np.float32)
    min_val = np.random.rand(1).astype(np.float32)
    max_val = np.random.rand(1).astype(np.float32)
    num_bits = 16
    narrow_range = False
    name = "fake_quant_test_5"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    gradients = np.random.rand(1, 2, 3, 4).astype(np.float32)
    inputs = np.random.rand(1, 2, 3, 4).astype(np.float32)
    min_val = np.random.rand(4).astype(np.float32)
    max_val = np.random.rand(4).astype(np.float32)
    num_bits = 6
    narrow_range = True
    name = "fake_quant_test_6"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    gradients = np.random.rand(6, 2).astype(np.float32)
    inputs = np.random.rand(6, 2).astype(np.float32)
    min_val = np.random.rand(2).astype(np.float32)
    max_val = np.random.rand(2).astype(np.float32)
    num_bits = 8
    narrow_range = False
    name = "fake_quant_test_7"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    gradients = np.random.rand(7, 8, 9).astype(np.float32)
    inputs = np.random.rand(7, 8, 9).astype(np.float32)
    min_val = np.random.rand(9).astype(np.float32)
    max_val = np.random.rand(9).astype(np.float32)
    num_bits = 10
    narrow_range = True
    name = "fake_quant_test_8"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    gradients = np.random.rand(2, 3).astype(np.float32)
    inputs = np.random.rand(2, 3).astype(np.float32)
    min_val = np.random.rand(3).astype(np.float32)
    max_val = np.random.rand(3).astype(np.float32)
    num_bits = 14
    narrow_range = False
    name = "fake_quant_test_9"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    gradients = np.random.rand(8).astype(np.float32)
    inputs = np.random.rand(8).astype(np.float32)
    min_val = np.random.rand(8).astype(np.float32)
    max_val = np.random.rand(8).astype(np.float32)
    num_bits = 5
    narrow_range = True
    name = "fake_quant_test_10"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
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
