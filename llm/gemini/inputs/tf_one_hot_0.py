
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_one_hot_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "indices": np.array([0, 1, 2], dtype=np.int32),
        "depth": 3,
        "on_value": np.array(1.0, dtype=np.float32),
        "off_value": np.array(0.0, dtype=np.float32),
        "axis": -1,
        "dtype": np.float32,
        "name": "one_hot_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "indices": np.array([[0, 2], [1, -1]], dtype=np.int32),
        "depth": 3,
        "on_value": np.array(1.0, dtype=np.float32),
        "off_value": np.array(0.0, dtype=np.float32),
        "axis": -1,
        "dtype": np.float32,
        "name": "one_hot_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "indices": np.array([1, 0], dtype=np.int32),
        "depth": 2,
        "on_value": np.array(5.0, dtype=np.float64),
        "off_value": np.array(-1.0, dtype=np.float64),
        "axis": 0,
        "dtype": np.float64,
        "name": "one_hot_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "indices": np.array([[[0, 1], [2, 3]]], dtype=np.int64),
        "depth": 4,
        "on_value": np.array(1, dtype=np.int32),
        "off_value": np.array(0, dtype=np.int32),
        "axis": -1,
        "dtype": np.int32,
        "name": "one_hot_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "indices": np.array(2, dtype=np.int32),
        "depth": 5,
        "on_value": np.array(1.0, dtype=np.float32),
        "off_value": np.array(0.0, dtype=np.float32),
        "axis": -1,
        "dtype": np.float32,
        "name": "one_hot_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "indices": np.array([0, -1, 2], dtype=np.int32),
        "depth": 3,
        "on_value": np.array(True, dtype=np.bool_),
        "off_value": np.array(False, dtype=np.bool_),
        "axis": -1,
        "dtype": np.bool_,
        "name": "one_hot_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "indices": np.array([[1], [0]], dtype=np.int32),
        "depth": 2,
        "on_value": np.array(2.5, dtype=np.float32),
        "off_value": np.array(0.1, dtype=np.float32),
        "axis": 1,
        "dtype": np.float32,
        "name": "one_hot_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "indices": np.array([2, 1, 0], dtype=np.int32),
        "depth": 3,
        "on_value": np.array(1, dtype=np.int64),
        "off_value": np.array(0, dtype=np.int64),
        "axis": 0,
        "dtype": np.int64,
        "name": "one_hot_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "indices": np.array([[[1]]], dtype=np.int32),
        "depth": 2,
        "on_value": np.array(10.0, dtype=np.float32),
        "off_value": np.array(-10.0, dtype=np.float32),
        "axis": 2,
        "dtype": np.float32,
        "name": "one_hot_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "indices": np.array([0], dtype=np.int32),
        "depth": 1,
        "on_value": np.array(1, dtype=np.int32),
        "off_value": np.array(0, dtype=np.int32),
        "axis": -1,
        "dtype": np.int32,
        "name": "one_hot_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.one_hot"] = tf_one_hot_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.one_hot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.one_hot'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.one_hot', generated_inputs['tf.one_hot'], lib="tf", suffix=0)
