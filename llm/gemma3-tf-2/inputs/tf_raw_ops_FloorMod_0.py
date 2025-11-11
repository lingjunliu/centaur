
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_FloorMod_inputs():
    list_of_inputs = []

    x = np.array([10, 20, 30], dtype=np.int32)
    y = np.array([3, 7, 2], dtype=np.int32)
    input_dict = {'name': 'floor_mod_1', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-10, -20, -30], dtype=np.int32)
    y = np.array([3, 7, 2], dtype=np.int32)
    input_dict = {'name': 'floor_mod_2', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10.5, 20.2, 30.9], dtype=np.float32)
    y = np.array([3.1, 7.5, 2.0], dtype=np.float32)
    input_dict = {'name': 'floor_mod_3', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-10, 20], [30, -40]], dtype=np.int64)
    y = np.array([3, 7], dtype=np.int64)
    input_dict = {'name': 'floor_mod_4', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.int8)
    y = np.array([2, 3, 4], dtype=np.int8)
    input_dict = {'name': 'floor_mod_5', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([100, 200, 300], dtype=np.float64)
    y = np.array([3.5, 7.1, 2.8], dtype=np.float64)
    input_dict = {'name': 'floor_mod_6', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, 20, 30], dtype=np.int16)
    y = np.array([3, 7, 2], dtype=np.int16)
    input_dict = {'name': 'floor_mod_7', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([2.0, 3.0, 4.0], dtype=np.float16)
    input_dict = {'name': 'floor_mod_8', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FloorMod"] = tf_raw_ops_FloorMod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.FloorMod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FloorMod'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.FloorMod', generated_inputs['tf.raw_ops.FloorMod'], lib="tf", suffix=0)
