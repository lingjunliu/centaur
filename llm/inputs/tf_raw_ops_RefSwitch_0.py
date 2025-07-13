
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ref_switch_inputs():
    list_of_inputs = []

    # Input 1: Simple case with True pred
    data = np.array([1, 2, 3], dtype=np.int32)
    pred = np.array(True, dtype=np.bool_)
    name = "switch_true"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple case with False pred
    data = np.array([4, 5, 6], dtype=np.int32)
    pred = np.array(False, dtype=np.bool_)
    name = "switch_false"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float data type
    data = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    pred = np.array(True, dtype=np.bool_)
    name = "switch_float"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Bool data type
    data = np.array([True, False, True], dtype=np.bool_)
    pred = np.array(False, dtype=np.bool_)
    name = "switch_bool"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D data
    data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    pred = np.array(True, dtype=np.bool_)
    name = "switch_2d"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D data
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    pred = np.array(False, dtype=np.bool_)
    name = "switch_3d"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty data
    data = np.array([], dtype=np.int32)
    pred = np.array(True, dtype=np.bool_)
    name = "switch_empty"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger data
    data = np.random.rand(100).astype(np.float32)
    pred = np.array(False, dtype=np.bool_)
    name = "switch_large"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative values in data
    data = np.array([-1, -2, -3], dtype=np.int32)
    pred = np.array(True, dtype=np.bool_)
    name = "switch_negative"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Zero value in data
    data = np.array([0, 1, 2], dtype=np.int32)
    pred = np.array(False, dtype=np.bool_)
    name = "switch_zero"
    input_dict = {"data": data, "pred": pred, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs
generated_inputs = {}
generated_inputs["tf.raw_ops.RefSwitch"] = tf_raw_ops_ref_switch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RefSwitch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RefSwitch'.")

check_valid('tf.raw_ops.RefSwitch', generated_inputs['tf.raw_ops.RefSwitch'], lib="tf", suffix=0)
