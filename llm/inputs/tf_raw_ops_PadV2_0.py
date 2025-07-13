
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_PadV2_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1, 1], [2, 2]], dtype=np.int32)
    paddings_tensor = np.array([[1, 1], [2, 2]], dtype=np.int32)
    constant_values_tensor = np.array(0, dtype=np.int32)
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "constant_values": constant_values_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([1, 2, 3], dtype=np.float32)
    paddings_tensor = np.array([[2, 1]], dtype=np.int32)
    constant_values_tensor = np.array(2.5, dtype=np.float32)
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "constant_values": constant_values_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    paddings_tensor = np.array([[0, 0], [1, 1], [2, 0]], dtype=np.int32)
    constant_values_tensor = np.array(-1, dtype=np.int64)
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "constant_values": constant_values_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    paddings_tensor = np.array([[0, 2]], dtype=np.int64)
    constant_values_tensor = np.array(10.0, dtype=np.float64)
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "constant_values": constant_values_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    paddings_tensor = np.array([[0, 0], [0, 0]], dtype=np.int32)
    constant_values_tensor = np.array(5, dtype=np.int32)
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "constant_values": constant_values_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([1, 2, 3], dtype=np.int64)
    paddings_tensor = np.array([[2, 0]], dtype=np.int64)
    constant_values_tensor = np.array(0, dtype=np.int64)
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "constant_values": constant_values_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    paddings_tensor = np.array([[1, 0], [0, 1]], dtype=np.int32)
    constant_values_tensor = np.array(-2.0, dtype=np.float32)
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "constant_values": constant_values_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    paddings_tensor = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int32)
    constant_values_tensor = np.array(100, dtype=np.int32)
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "constant_values": constant_values_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([1, 2, 3], dtype=np.float64)
    paddings_tensor = np.array([[0, 0]], dtype=np.int32)
    constant_values_tensor = np.array(1.0, dtype=np.float64)
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "constant_values": constant_values_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    paddings_tensor = np.array([[2, 1], [1, 2]], dtype=np.int64)
    constant_values_tensor = np.array(-1, dtype=np.int32)
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "constant_values": constant_values_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.PadV2"] = tf_raw_ops_PadV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.PadV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.PadV2'.")

check_valid('tf.raw_ops.PadV2', generated_inputs['tf.raw_ops.PadV2'], lib="tf", suffix=0)
