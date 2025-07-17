
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_switch_inputs():
    list_of_inputs = []

    # Input 1: Basic case with True predicate
    data = np.array([1, 2, 3], dtype=np.int32)
    pred = np.array(True, dtype=np.bool_)
    name = "switch_1"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic case with False predicate
    data = np.array([4, 5, 6], dtype=np.float32)
    pred = np.array(False, dtype=np.bool_)
    name = "switch_2"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional data with True predicate
    data = np.array([[1, 2], [3, 4]], dtype=np.int64)
    pred = np.array(True, dtype=np.bool_)
    name = "switch_3"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional data with False predicate
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    pred = np.array(False, dtype=np.bool_)
    name = "switch_4"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty data array with True predicate
    data = np.array([], dtype=np.int32)
    pred = np.array(True, dtype=np.bool_)
    name = "switch_5"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty data array with False predicate
    data = np.array([], dtype=np.float32)
    pred = np.array(False, dtype=np.bool_)
    name = "switch_6"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Data with string dtype and True predicate - Removed str_ and unicode_ types as they might not be directly supported
    data = np.array(["hello", "world"], dtype=np.object_)
    pred = np.array(True, dtype=np.bool_)
    name = "switch_7"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Data with string dtype and False predicate - Removed str_ and unicode_ types as they might not be directly supported
    data = np.array(["foo", "bar"], dtype=np.object_)
    pred = np.array(False, dtype=np.bool_)
    name = "switch_8"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large data array with True predicate
    data = np.random.rand(100, 100).astype(np.float32)
    pred = np.array(True, dtype=np.bool_)
    name = "switch_9"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large data array with False predicate
    data = np.random.randint(0, 10, size=(50, 50, 50)).astype(np.int32)
    pred = np.array(False, dtype=np.bool_)
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
