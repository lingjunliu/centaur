
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FakeQuantWithMinMaxArgs_inputs():
    list_of_inputs = []

    # Input 1
    inputs = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    min_val = -2.0
    max_val = 2.0
    num_bits = 8
    narrow_range = False
    name = "test_quant1"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    min_val = -1.0
    max_val = 3.0
    num_bits = 4
    narrow_range = True
    name = None
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = np.array([0.5, 1.5, -0.5], dtype=np.float32)
    min_val = -1.5
    max_val = 2.5
    num_bits = 16
    narrow_range = False
    name = "quant3"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    min_val = 0.0
    max_val = 1.0
    num_bits = 2
    narrow_range = True
    name = None
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=np.float32)
    min_val = -5.0
    max_val = 5.0
    num_bits = 8
    narrow_range = False
    name = "quant5"
    input_dict = {"inputs": inputs, "min": min_val, "max": max_val, "num_bits": num_bits, "narrow_range": narrow_range, "name": name}
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
