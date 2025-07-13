
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FakeQuantWithMinMaxVarsPerChannel_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive values
    inputs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    min_val = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    max_val = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "basic_positive"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Case with negative values
    inputs = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
    min_val = np.array([-4.0, -5.0, -6.0], dtype=np.float32)
    max_val = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "basic_negative"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Case with mixed positive and negative values
    inputs = np.array([[-1.0, 0.0, 1.0], [-2.0, 2.0, 3.0]], dtype=np.float32)
    min_val = np.array([-2.0, -1.0, 0.0], dtype=np.float32)
    max_val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "mixed_values"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different num_bits
    inputs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    min_val = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    max_val = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    num_bits = 4
    narrow_range = False
    name = "different_num_bits"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: narrow_range = True
    inputs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    min_val = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    max_val = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    num_bits = 8
    narrow_range = True
    name = "narrow_range_true"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D input tensor - Fixed min/max calculation
    inputs = np.random.rand(2, 3, 4).astype(np.float32)
    min_val = np.array([np.min(inputs[..., i]) for i in range(inputs.shape[-1])], dtype=np.float32)
    max_val = np.array([np.max(inputs[..., i]) for i in range(inputs.shape[-1])], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "3d_input"
    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different min/max ranges
    inputs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    min_val = np.array([-1.0, 0.5, 1.5], dtype=np.float32)
    max_val = np.array([3.0, 4.5, 5.5], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "different_min_max"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: num_bits = 2
    inputs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    min_val = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    max_val = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    num_bits = 2
    narrow_range = False
    name = "num_bits_2"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: num_bits = 16
    inputs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    min_val = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    max_val = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    num_bits = 16
    narrow_range = False
    name = "num_bits_16"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D input tensor
    inputs = np.random.rand(5).astype(np.float32)
    min_val = np.array([np.min(inputs)], dtype=np.float32)
    max_val = np.array([np.max(inputs)], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "1d_input"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: min > max
    inputs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    min_val = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    max_val = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "min_greater_than_max"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: empty input
    inputs = np.array([], dtype=np.float32)
    min_val = np.array([], dtype=np.float32)
    max_val = np.array([], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "empty_input"
    input_dict = {
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
generated_inputs["tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel"] = tf_raw_ops_FakeQuantWithMinMaxVarsPerChannel_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel'.")

check_valid('tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel', generated_inputs['tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel'], lib="tf", suffix=0)
