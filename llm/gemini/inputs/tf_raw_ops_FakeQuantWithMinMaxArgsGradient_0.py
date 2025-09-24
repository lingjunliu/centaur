
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FakeQuantWithMinMaxArgsGradient_inputs():
    list_of_inputs = []

    # Input 1
    gradients = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    inputs = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": -1.0,
        "max": 4.0,
        "num_bits": 8,
        "narrow_range": False,
        "name": "test_op_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    gradients = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32)
    inputs = np.array([-0.5, -1.5, -2.5, -3.5], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": -4.0,
        "max": 1.0,
        "num_bits": 8,
        "narrow_range": False,
        "name": "test_op_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    gradients = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    inputs = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": -2.0,
        "max": 5.0,
        "num_bits": 8,
        "narrow_range": False,
        "name": "test_op_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    gradients = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    inputs = np.array([[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": -3.0,
        "max": 6.0,
        "num_bits": 8,
        "narrow_range": False,
        "name": "test_op_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    gradients = np.array([1.0], dtype=np.float32)
    inputs = np.array([0.5], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": 0.0,
        "max": 1.0,
        "num_bits": 8,
        "narrow_range": True,
        "name": "test_op_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    gradients = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    inputs = np.array([-0.5, 1.5, -2.5, 3.5], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": -3.5,
        "max": 3.5,
        "num_bits": 4,
        "narrow_range": True,
        "name": "test_op_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    gradients = np.zeros((2, 2), dtype=np.float32)
    inputs = np.ones((2, 2), dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": -1.0,
        "max": 1.0,
        "num_bits": 16,
        "narrow_range": True,
        "name": "test_op_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    gradients = np.random.rand(3, 3).astype(np.float32)
    inputs = np.random.rand(3, 3).astype(np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": -2.0,
        "max": 2.0,
        "num_bits": 8,
        "narrow_range": False,
        "name": "test_op_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    gradients = np.full((1, 5), 0.5, dtype=np.float32)
    inputs = np.full((1, 5), 0.25, dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": 0.0,
        "max": 0.5,
        "num_bits": 8,
        "narrow_range": False,
        "name": "test_op_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    gradients = np.eye(4, dtype=np.float32)
    inputs = np.eye(4, dtype=np.float32) * 0.75
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": 0.0,
        "max": 1.0,
        "num_bits": 8,
        "narrow_range": False,
        "name": "test_op_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FakeQuantWithMinMaxArgsGradient"] = tf_raw_ops_FakeQuantWithMinMaxArgsGradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FakeQuantWithMinMaxArgsGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FakeQuantWithMinMaxArgsGradient'.")

check_valid('tf.raw_ops.FakeQuantWithMinMaxArgsGradient', generated_inputs['tf.raw_ops.FakeQuantWithMinMaxArgsGradient'], lib="tf", suffix=0)
