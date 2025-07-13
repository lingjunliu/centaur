
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FakeQuantWithMinMaxArgs_inputs():
    list_of_inputs = []

    # Input 1
    inputs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    min_val = -1.0
    max_val = 4.0
    num_bits_val = 8
    narrow_range_val = False
    name_val = "test_quant1"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits_val,
        "narrow_range": narrow_range_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    min_val = -4.0
    max_val = -0.5
    num_bits_val = 10
    narrow_range_val = True
    name_val = "test_quant2"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits_val,
        "narrow_range": narrow_range_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    min_val = 0.0
    max_val = 5.0
    num_bits_val = 2
    narrow_range_val = False
    name_val = "test_quant3"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits_val,
        "narrow_range": narrow_range_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    min_val = -2.0
    max_val = 8.0
    num_bits_val = 16
    narrow_range_val = True
    name_val = "test_quant4"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits_val,
        "narrow_range": narrow_range_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    min_val = -0.5
    max_val = 1.5
    num_bits_val = 4
    narrow_range_val = False
    name_val = "test_quant5"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits_val,
        "narrow_range": narrow_range_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    min_val = -5.0
    max_val = 5.0
    num_bits_val = 8
    narrow_range_val = True
    name_val = "test_quant6"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits_val,
        "narrow_range": narrow_range_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    min_val = 0.0
    max_val = 1.0
    num_bits_val = 3
    narrow_range_val = False
    name_val = "test_quant7"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits_val,
        "narrow_range": narrow_range_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    inputs = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
    min_val = -10.0
    max_val = 10.0
    num_bits_val = 12
    narrow_range_val = True
    name_val = "test_quant8"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits_val,
        "narrow_range": narrow_range_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = np.array([0.0], dtype=np.float32)
    min_val = -1.0
    max_val = 1.0
    num_bits_val = 6
    narrow_range_val = False
    name_val = "test_quant9"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits_val,
        "narrow_range": narrow_range_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    inputs = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    min_val = -0.5
    max_val = 0.5
    num_bits_val = 5
    narrow_range_val = True
    name_val = "test_quant10"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits_val,
        "narrow_range": narrow_range_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FakeQuantWithMinMaxArgs"] = tf_raw_ops_FakeQuantWithMinMaxArgs_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FakeQuantWithMinMaxArgs' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FakeQuantWithMinMaxArgs'.")

check_valid('tf.raw_ops.FakeQuantWithMinMaxArgs', generated_inputs['tf.raw_ops.FakeQuantWithMinMaxArgs'], lib="tf", suffix=0)
