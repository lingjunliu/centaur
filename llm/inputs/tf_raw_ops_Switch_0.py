
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_switch_inputs():
    list_of_inputs = []

    # Input 1: Basic case, True
    data = np.array([1, 2, 3], dtype=np.int32)
    pred = np.array(True, dtype=np.bool_)
    name = "switch_1"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic case, False
    data = np.array([4, 5, 6], dtype=np.int32)
    pred = np.array(False, dtype=np.bool_)
    name = "switch_2"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float data
    data = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    pred = np.array(True, dtype=np.bool_)
    name = "switch_3"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional data
    data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    pred = np.array(False, dtype=np.bool_)
    name = "switch_4"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Bool data
    data = np.array([True, False, True], dtype=np.bool_)
    pred = np.array(True, dtype=np.bool_)
    name = "switch_5"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty data array
    data = np.array([], dtype=np.int32)
    pred = np.array(False, dtype=np.bool_)
    name = "switch_6"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: String data
    data = np.array(["a", "b", "c"], dtype=np.string_)
    pred = np.array(True, dtype=np.bool_)
    name = "switch_7"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: Larger array, different name
    data = np.random.rand(10, 10).astype(np.float32)
    pred = np.array(False, dtype=np.bool_)
    name = "another_switch_name"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Int64 data type
    data = np.array([1, 2, 3], dtype=np.int64)
    pred = np.array(True, dtype=np.bool_)
    name = "switch_9"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex data, different pred
    data = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    pred = np.array(True, dtype=np.bool_)
    name = "switch_10"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Switch"] = tf_raw_ops_switch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Switch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Switch'.")

check_valid('tf.raw_ops.Switch', generated_inputs['tf.raw_ops.Switch'], lib="tf", suffix=0)
