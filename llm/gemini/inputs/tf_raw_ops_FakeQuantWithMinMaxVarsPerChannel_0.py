
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FakeQuantWithMinMaxVarsPerChannel_inputs():
    list_of_inputs = []

    # Input 1
    inputs = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.5, 1.5]], dtype=np.float32)
    min_val = np.array([-2.0, -1.0, 0.0], dtype=np.float32)
    max_val = np.array([2.0, 1.0, 2.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "fake_quant1"
    input_dict = {"num_bits": num_bits, "narrow_range": narrow_range, "name": name, "inputs": inputs, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    min_val = np.array([0.0, 0.1], dtype=np.float32)
    max_val = np.array([1.0, 1.1], dtype=np.float32)
    num_bits = 4
    narrow_range = True
    name = "fake_quant2"
    input_dict = {"num_bits": num_bits, "narrow_range": narrow_range, "name": name, "inputs": inputs, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    min_val = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    max_val = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    num_bits = 2
    narrow_range = False
    name = "fake_quant3"
    input_dict = {"num_bits": num_bits, "narrow_range": narrow_range, "name": name, "inputs": inputs, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4
    inputs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    min_val = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    max_val = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    num_bits = 16
    narrow_range = True
    name = "fake_quant4"
    input_dict = {"num_bits": num_bits, "narrow_range": narrow_range, "name": name, "inputs": inputs, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = np.array([[-0.5, 0.5], [-1.0, 1.0]], dtype=np.float32)
    min_val = np.array([-1.0, 0.0], dtype=np.float32)
    max_val = np.array([0.0, 1.0], dtype=np.float32)
    num_bits = 5
    narrow_range = False
    name = "fake_quant5"
    input_dict = {"num_bits": num_bits, "narrow_range": narrow_range, "name": name, "inputs": inputs, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    min_val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    max_val = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    num_bits = 10
    narrow_range = True
    name = "fake_quant6"
    input_dict = {"num_bits": num_bits, "narrow_range": narrow_range, "name": name, "inputs": inputs, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = np.array([[-1.0, -0.5], [0.0, 0.5]], dtype=np.float32)
    min_val = np.array([-2.0, -1.0], dtype=np.float32)
    max_val = np.array([1.0, 2.0], dtype=np.float32)
    num_bits = 6
    narrow_range = False
    name = "fake_quant7"
    input_dict = {"num_bits": num_bits, "narrow_range": narrow_range, "name": name, "inputs": inputs, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    inputs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    min_val = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    max_val = np.array([0.3, 0.4, 0.5], dtype=np.float32)
    num_bits = 7
    narrow_range = True
    name = "fake_quant8"
    input_dict = {"num_bits": num_bits, "narrow_range": narrow_range, "name": name, "inputs": inputs, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    min_val = np.array([-3.0, -3.0, -3.0, -3.0, -3.0], dtype=np.float32)
    max_val = np.array([3.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float32)
    num_bits = 9
    narrow_range = False
    name = "fake_quant9"
    input_dict = {"num_bits": num_bits, "narrow_range": narrow_range, "name": name, "inputs": inputs, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    inputs = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    min_val = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    max_val = np.array([2.0, 2.5, 3.0], dtype=np.float32)
    num_bits = 11
    narrow_range = True
    name = "fake_quant10"
    input_dict = {"num_bits": num_bits, "narrow_range": narrow_range, "name": name, "inputs": inputs, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    inputs = np.array([[-0.75, -0.25], [0.25, 0.75]], dtype=np.float32)
    min_val = np.array([-1.0, -0.5], dtype=np.float32)
    max_val = np.array([0.5, 1.0], dtype=np.float32)
    num_bits = 12
    narrow_range = False
    name = "fake_quant11"
    input_dict = {"num_bits": num_bits, "narrow_range": narrow_range, "name": name, "inputs": inputs, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    inputs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    min_val = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    max_val = np.array([0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    num_bits = 13
    narrow_range = True
    name = "fake_quant12"
    input_dict = {"num_bits": num_bits, "narrow_range": narrow_range, "name": name, "inputs": inputs, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel"] = tf_raw_ops_FakeQuantWithMinMaxVarsPerChannel_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel'.")

check_valid('tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel', generated_inputs['tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel'], lib="tf", suffix=0)
