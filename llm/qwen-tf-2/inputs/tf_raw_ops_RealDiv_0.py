
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def generate_realdiv_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensors
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    input_dict = {
        "name": "test1",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 tensors
    x = np.array([10.5, 20.7], dtype=np.float64)
    y = np.array([2.5, 4.0], dtype=np.float64)
    input_dict = {
        "name": "test2",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32 tensors with negative values
    x = np.array([-1.0, -2.0], dtype=np.float32)
    y = np.array([2.0, -4.0], dtype=np.float32)
    input_dict = {
        "name": "test3",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: scalar float32 tensors
    x = np.array(1.0, dtype=np.float32)
    y = np.array(2.0, dtype=np.float32)
    input_dict = {
        "name": "test4",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 tensors with broadcasting
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y = np.array([2.0], dtype=np.float64)
    input_dict = {
        "name": "test5",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: half tensors
    x = np.array([1.0, 2.0], dtype=np.float16)
    y = np.array([2.0, 4.0], dtype=np.float16)
    input_dict = {
        "name": "test6",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex64 tensors
    x = np.array([1+2j, 3+4j], dtype=np.complex64)
    y = np.array([2+1j, 2+2j], dtype=np.complex64)
    input_dict = {
        "name": "test7",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32 tensors with different dimensions
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    y = np.array([[[2.0, 4.0], [6.0, 8.0]], [[10.0, 12.0], [14.0, 16.0]]], dtype=np.float32)
    input_dict = {
        "name": "test8",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32 tensors with broadcasting
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    y = np.array([[[2.0, 4.0], [6.0, 8.0]]], dtype=np.float32)
    input_dict = {
        "name": "test9",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 tensors with various shapes
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    y = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    input_dict = {
        "name": "test10",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.RealDiv"] = generate_realdiv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RealDiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RealDiv'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.RealDiv', generated_inputs['tf.raw_ops.RealDiv'], lib="tf", suffix=0)
