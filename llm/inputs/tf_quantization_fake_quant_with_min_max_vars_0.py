
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_quantization_fake_quant_with_min_max_vars_inputs():
    list_of_inputs = []

    # Input 1: Basic test
    inputs = tf.constant([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=tf.float32)
    min_val = tf.constant(-1.0, dtype=tf.float32)
    max_val = tf.constant(1.0, dtype=tf.float32)
    num_bits = 8
    narrow_range = False
    name = "basic_test"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different range
    inputs = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=tf.float32)
    min_val = tf.constant(-2.0, dtype=tf.float32)
    max_val = tf.constant(2.0, dtype=tf.float32)
    num_bits = 8
    narrow_range = False
    name = "different_range"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Narrow range
    inputs = tf.constant([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=tf.float32)
    min_val = tf.constant(-1.0, dtype=tf.float32)
    max_val = tf.constant(1.0, dtype=tf.float32)
    num_bits = 8
    narrow_range = True
    name = "narrow_range"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different num_bits
    inputs = tf.constant([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=tf.float32)
    min_val = tf.constant(-1.0, dtype=tf.float32)
    max_val = tf.constant(1.0, dtype=tf.float32)
    num_bits = 4
    narrow_range = False
    name = "different_num_bits"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D input
    inputs = tf.constant([[-1.0, -0.5], [0.0, 0.5], [1.0, 1.5]], dtype=tf.float32)
    min_val = tf.constant(-1.0, dtype=tf.float32)
    max_val = tf.constant(1.5, dtype=tf.float32)
    num_bits = 8
    narrow_range = False
    name = "2d_input"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D input
    inputs = tf.constant([[[1.2, -0.3], [0.7, 2.1]], [[0.5, -1.0], [0.0, 1.0]]], dtype=tf.float32)
    min_val = tf.constant(-1.0, dtype=tf.float32)
    max_val = tf.constant(2.1, dtype=tf.float32)
    num_bits = 8
    narrow_range = False
    name = "3d_input"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: all positive, min > 0
    inputs = tf.constant([0.1, 0.5, 1.0, 1.5, 2.0], dtype=tf.float32)
    min_val = tf.constant(0.1, dtype=tf.float32)
    max_val = tf.constant(2.0, dtype=tf.float32)
    num_bits = 8
    narrow_range = False
    name = "all_positive"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: all negative, max < 0
    inputs = tf.constant([-2.0, -1.5, -1.0, -0.5, -0.1], dtype=tf.float32)
    min_val = tf.constant(-2.0, dtype=tf.float32)
    max_val = tf.constant(-0.1, dtype=tf.float32)
    num_bits = 8
    narrow_range = False
    name = "all_negative"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: num_bits = 2
    inputs = tf.constant([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=tf.float32)
    min_val = tf.constant(-1.0, dtype=tf.float32)
    max_val = tf.constant(1.0, dtype=tf.float32)
    num_bits = 2
    narrow_range = False
    name = "num_bits_2"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: min == max
    inputs = tf.constant([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=tf.float32)
    min_val = tf.constant(0.5, dtype=tf.float32)
    max_val = tf.constant(0.5, dtype=tf.float32)
    num_bits = 8
    narrow_range = False
    name = "min_equals_max"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Convert tf.constant to numpy arrays
    for input_dict in list_of_inputs:
        input_dict['inputs'] = input_dict['inputs'].numpy()
        input_dict['min'] = input_dict['min'].numpy()
        input_dict['max'] = input_dict['max'].numpy()

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.quantization.fake_quant_with_min_max_vars"] = tf_quantization_fake_quant_with_min_max_vars_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.quantization.fake_quant_with_min_max_vars' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.quantization.fake_quant_with_min_max_vars'.")

check_valid('tf.quantization.fake_quant_with_min_max_vars', generated_inputs['tf.quantization.fake_quant_with_min_max_vars'], lib="tf", suffix=0)
