
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FakeQuantWithMinMaxVarsGradient_inputs():
    list_of_inputs = []

    # Input 1: Basic test case
    gradients = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    inputs = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(3.0, dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test1"

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

    # Input 2: Test with different values and narrow_range=True
    gradients = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    inputs = np.array([-0.5, -1.5, -2.5], dtype=np.float32)
    min_val = np.array(-3.0, dtype=np.float32)
    max_val = np.array(0.0, dtype=np.float32)
    num_bits = 4
    narrow_range = True
    name = "test2"

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

    # Input 3: Test with higher dimensions
    gradients = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    inputs = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(4.0, dtype=np.float32)
    num_bits = 6
    narrow_range = False
    name = "test3"

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

    # Input 4: Test with negative min and max
    gradients = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    inputs = np.array([-1.5, -0.5, 0.5], dtype=np.float32)
    min_val = np.array(-2.0, dtype=np.float32)
    max_val = np.array(1.0, dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test4"

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

    # Input 5: Test with small range
    gradients = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    inputs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(0.5, dtype=np.float32)
    num_bits = 7
    narrow_range = False
    name = "test5"

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

    # Input 6: Test with large gradients
    gradients = np.array([1000.0, 2000.0, 3000.0], dtype=np.float32)
    inputs = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(3.0, dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test6"

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

    # Input 7: Test with 2 bits
    gradients = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    inputs = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(3.0, dtype=np.float32)
    num_bits = 2
    narrow_range = False
    name = "test7"

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

    # Input 8: Test with 8 bits and narrow range
    gradients = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    inputs = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(3.0, dtype=np.float32)
    num_bits = 8
    narrow_range = True
    name = "test8"

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

    # Input 9: Test with all negative values
    gradients = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    inputs = np.array([-0.5, -1.5, -2.5], dtype=np.float32)
    min_val = np.array(-3.0, dtype=np.float32)
    max_val = np.array(-0.0, dtype=np.float32)
    num_bits = 5
    narrow_range = False
    name = "test9"

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

    # Input 10: Test with zero gradients
    gradients = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    inputs = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(3.0, dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test10"

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

generated_inputs = {}
generated_inputs["tf.raw_ops.FakeQuantWithMinMaxVarsGradient"] = tf_raw_ops_FakeQuantWithMinMaxVarsGradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FakeQuantWithMinMaxVarsGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FakeQuantWithMinMaxVarsGradient'.")

check_valid('tf.raw_ops.FakeQuantWithMinMaxVarsGradient', generated_inputs['tf.raw_ops.FakeQuantWithMinMaxVarsGradient'], lib="tf", suffix=0)
