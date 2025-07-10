
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_quantization_fake_quant_with_min_max_args_inputs():
    list_of_inputs = []

    # Input 1
    inputs = tf.constant([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    min_val = -2.0
    max_val = 4.0
    num_bits = 8
    narrow_range = False
    name = "fake_quant1"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = tf.constant([[-1.0, 0.0], [1.0, 2.0]], dtype=tf.float32).numpy()
    min_val = -1.0
    max_val = 1.0
    num_bits = 4
    narrow_range = True
    name = "fake_quant2"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = tf.constant([0.0, 0.5, 1.0, 1.5, 2.0], dtype=tf.float32).numpy()
    min_val = 0.0
    max_val = 2.0
    num_bits = 16
    narrow_range = False
    name = "fake_quant3"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = tf.constant([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=tf.float32).numpy()
    min_val = -5.0
    max_val = 5.0
    num_bits = 2
    narrow_range = True
    name = "fake_quant4"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=tf.float32).numpy()
    min_val = 1.0
    max_val = 8.0
    num_bits = 8
    narrow_range = False
    name = "fake_quant5"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = tf.constant([-0.5, 0.2, 0.7, 1.2], dtype=tf.float32).numpy()
    min_val = -1.0
    max_val = 2.0
    num_bits = 6
    narrow_range = True
    name = "fake_quant6"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    min_val = 1.0
    max_val = 5.0
    num_bits = 10
    narrow_range = False
    name = "fake_quant7"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    inputs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    min_val = -3.0
    max_val = 3.0
    num_bits = 12
    narrow_range = True
    name = "fake_quant8"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).reshape((2,2))
    min_val = 0.0
    max_val = 1.0
    num_bits = 5
    narrow_range = False
    name = "fake_quant9"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    inputs = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
    min_val = -10.0
    max_val = 10.0
    num_bits = 3
    narrow_range = True
    name = "fake_quant10"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.quantization.fake_quant_with_min_max_args"] = tf_quantization_fake_quant_with_min_max_args_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.quantization.fake_quant_with_min_max_args' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.quantization.fake_quant_with_min_max_args'.")

check_valid('tf.quantization.fake_quant_with_min_max_args', generated_inputs['tf.quantization.fake_quant_with_min_max_args'], lib="tf", suffix=0)
