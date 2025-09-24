
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FakeQuantWithMinMaxArgs_inputs():
    list_of_inputs = []

    # Input 1: Basic test case
    inputs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    min_val = -2.0
    max_val = 4.0
    num_bits_val = 8
    narrow_range_val = False
    name_val = "basic_test"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits_val, "narrow_range": narrow_range_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different range
    inputs = np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=np.float32)
    min_val = -5.0
    max_val = 5.0
    num_bits_val = 8
    narrow_range_val = False
    name_val = "different_range"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits_val, "narrow_range": narrow_range_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Narrow range
    inputs = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    min_val = 0.0
    max_val = 1.0
    num_bits_val = 8
    narrow_range_val = True
    name_val = "narrow_range"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits_val, "narrow_range": narrow_range_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: More bits
    inputs = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    min_val = -1.0
    max_val = 1.0
    num_bits_val = 16
    narrow_range_val = False
    name_val = "more_bits"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits_val, "narrow_range": narrow_range_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Fewer bits
    inputs = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    min_val = -1.0
    max_val = 1.0
    num_bits_val = 2
    narrow_range_val = False
    name_val = "fewer_bits"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits_val, "narrow_range": narrow_range_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D input
    inputs = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    min_val = -2.0
    max_val = 3.0
    num_bits_val = 8
    narrow_range_val = False
    name_val = "2d_input"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits_val, "narrow_range": narrow_range_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D input
    inputs = np.array([[[ -1.0, 0.0], [1.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]]], dtype=np.float32)
    min_val = -1.0
    max_val = 6.0
    num_bits_val = 8
    narrow_range_val = False
    name_val = "3d_input"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits_val, "narrow_range": narrow_range_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Only positive inputs
    inputs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    min_val = 1.0
    max_val = 3.0
    num_bits_val = 8
    narrow_range_val = False
    name_val = "positive_inputs"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits_val, "narrow_range": narrow_range_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Only negative inputs
    inputs = np.array([-3.0, -2.0, -1.0], dtype=np.float32)
    min_val = -3.0
    max_val = -1.0
    num_bits_val = 8
    narrow_range_val = False
    name_val = "negative_inputs"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits_val, "narrow_range": narrow_range_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Small min/max range
    inputs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    min_val = 0.1
    max_val = 0.3
    num_bits_val = 8
    narrow_range_val = False
    name_val = "small_range"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits_val, "narrow_range": narrow_range_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FakeQuantWithMinMaxArgs"] = tf_raw_ops_FakeQuantWithMinMaxArgs_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FakeQuantWithMinMaxArgs' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FakeQuantWithMinMaxArgs'.")

check_valid('tf.raw_ops.FakeQuantWithMinMaxArgs', generated_inputs['tf.raw_ops.FakeQuantWithMinMaxArgs'], lib="tf", suffix=0)
