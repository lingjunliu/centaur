
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FakeQuantWithMinMaxVarsPerChannelGradient_inputs():
    list_of_inputs = []

    # Input 1
    gradients = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    inputs = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    min_val = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    max_val = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test_op_1"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    gradients = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    inputs = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    min_val = np.array([0.0, -1.0], dtype=np.float32)
    max_val = np.array([1.0, 0.0], dtype=np.float32)
    num_bits = 4
    narrow_range = True
    name = "test_op_2"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    gradients = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    inputs = np.array([[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]], dtype=np.float32)
    min_val = np.array([0.0, 1.0], dtype=np.float32)
    max_val = np.array([1.0, 3.0], dtype=np.float32)
    num_bits = 2
    narrow_range = False
    name = "test_op_3"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    gradients = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    inputs = np.array([-0.5, -1.5, -2.5], dtype=np.float32)
    min_val = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    max_val = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    num_bits = 16
    narrow_range = True
    name = "test_op_4"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    gradients = np.array([[-1.5, -2.5], [-3.5, -4.5]], dtype=np.float32)
    inputs = np.array([[0.6, 0.7], [0.8, 0.9]], dtype=np.float32)
    min_val = np.array([0.5, 0.6], dtype=np.float32)
    max_val = np.array([1.5, 2.5], dtype=np.float32)
    num_bits = 5
    narrow_range = False
    name = "test_op_5"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    gradients = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    inputs = np.array([0.4, 1.6, 2.7], dtype=np.float32)
    min_val = np.array([0.1, 1.2, 2.3], dtype=np.float32)
    max_val = np.array([1.4, 3.5, 5.6], dtype=np.float32)
    num_bits = 7
    narrow_range = True
    name = "test_op_6"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    gradients = np.array([[-1.1, -2.2], [-3.3, -4.4]], dtype=np.float32)
    inputs = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    min_val = np.array([0.0, -0.1], dtype=np.float32)
    max_val = np.array([0.1, 0.0], dtype=np.float32)
    num_bits = 6
    narrow_range = False
    name = "test_op_7"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    gradients = np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]], dtype=np.float32)
    inputs = np.array([[[0.6, 1.7], [2.8, 3.9]], [[4.1, 5.2], [6.3, 7.4]]], dtype=np.float32)
    min_val = np.array([0.2, 1.3], dtype=np.float32)
    max_val = np.array([1.5, 3.6], dtype=np.float32)
    num_bits = 3
    narrow_range = True
    name = "test_op_8"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    gradients = np.array([1.6, 2.7, 3.8], dtype=np.float32)
    inputs = np.array([-0.6, -1.7, -2.8], dtype=np.float32)
    min_val = np.array([-1.1, -2.2, -3.3], dtype=np.float32)
    max_val = np.array([0.1, 1.2, 2.3], dtype=np.float32)
    num_bits = 15
    narrow_range = False
    name = "test_op_9"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    gradients = np.array([[-1.6, -2.7], [-3.8, -4.9]], dtype=np.float32)
    inputs = np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float32)
    min_val = np.array([0.6, 0.7], dtype=np.float32)
    max_val = np.array([1.6, 2.7], dtype=np.float32)
    num_bits = 9
    narrow_range = True
    name = "test_op_10"
    input_dict = {"gradients": gradients, "inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FakeQuantWithMinMaxVarsPerChannelGradient"] = tf_raw_ops_FakeQuantWithMinMaxVarsPerChannelGradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FakeQuantWithMinMaxVarsPerChannelGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FakeQuantWithMinMaxVarsPerChannelGradient'.")

check_valid('tf.raw_ops.FakeQuantWithMinMaxVarsPerChannelGradient', generated_inputs['tf.raw_ops.FakeQuantWithMinMaxVarsPerChannelGradient'], lib="tf", suffix=0)
