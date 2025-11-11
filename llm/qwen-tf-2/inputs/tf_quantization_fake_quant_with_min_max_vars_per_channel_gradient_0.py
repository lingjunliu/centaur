
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def generate_fake_quant_with_min_max_vars_per_channel_gradient_inputs():
    list_of_inputs = []

    # Input 1, valid
    gradients = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    min_tensor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test1"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    gradients = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    min_tensor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test2"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    gradients = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    min_tensor = np.array([0.0, 0.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test3"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    gradients = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [0.0, 1.0, 2.0]]], dtype=np.float32)
    input_tensor = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [0.0, 1.0, 2.0]]], dtype=np.float32)
    min_tensor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test4"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    gradients = np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    input_tensor = np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    min_tensor = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test5"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    gradients = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    min_tensor = np.array([-1.0, -1.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0], dtype=np.float32)
    num_bits = 8
    narrow_range = True
    name = "test6"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    gradients = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    min_tensor = np.array([-1.0, -1.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0], dtype=np.float32)
    num_bits = 16
    narrow_range = False
    name = "test7"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    gradients = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    min_tensor = np.array([-1.0, -1.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0], dtype=np.float32)
    num_bits = 16
    narrow_range = True
    name = "test8"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    gradients = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    min_tensor = np.array([-1.0, -1.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test9"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    gradients = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    min_tensor = np.array([-1.0, -1.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test10"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient"] = generate_fake_quant_with_min_max_vars_per_channel_gradient_inputs()

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
