
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_RefSwitch_inputs():
    list_of_inputs = []

    # Input 1
    data = tf.Variable(np.array([1, 2, 3], dtype=np.int32))
    pred = tf.constant(True, dtype=tf.bool)
    name = "switch_1"

    input_dict = {
        "data": data.numpy(),
        "pred": pred,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.float32))
    pred = tf.constant(False, dtype=tf.bool)
    name = "switch_2"

    input_dict = {
        "data": data.numpy(),
        "pred": pred,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = tf.Variable(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64))
    pred = tf.constant(True, dtype=tf.bool)
    name = "switch_3"

    input_dict = {
        "data": data.numpy(),
        "pred": pred,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = tf.Variable(np.array([-1, -2, -3], dtype=np.int32))
    pred = tf.constant(False, dtype=tf.bool)
    name = "switch_4"

    input_dict = {
        "data": data.numpy(),
        "pred": pred,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = tf.Variable(np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64))
    pred = tf.constant(True, dtype=tf.bool)
    name = "switch_5"

    input_dict = {
        "data": data.numpy(),
        "pred": pred,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = tf.Variable(np.array(5, dtype=np.int32))
    pred = tf.constant(False, dtype=tf.bool)
    name = "switch_6"

    input_dict = {
        "data": data.numpy(),
        "pred": pred,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    data = tf.Variable(np.array([1, 2, 3, 4, 5], dtype=np.int32))
    pred = tf.constant(True, dtype=tf.bool)
    name = "switch_7"

    input_dict = {
        "data": data.numpy(),
        "pred": pred,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    data = tf.Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    pred = tf.constant(False, dtype=tf.bool)
    name = "switch_8"

    input_dict = {
        "data": data.numpy(),
        "pred": pred,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    data = tf.Variable(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32))
    pred = tf.constant(True, dtype=tf.bool)
    name = "switch_9"

    input_dict = {
        "data": data.numpy(),
        "pred": pred,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    data = tf.Variable(np.array([1], dtype=np.int32))
    pred = tf.constant(False, dtype=tf.bool)
    name = "switch_10"

    input_dict = {
        "data": data.numpy(),
        "pred": pred,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RefSwitch"] = tf_raw_ops_RefSwitch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RefSwitch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RefSwitch'.")

check_valid('tf.raw_ops.RefSwitch', generated_inputs['tf.raw_ops.RefSwitch'], lib="tf", suffix=0)
