
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_LRN_inputs():
    list_of_inputs = []

    input_1 = np.random.rand(2, 4, 4, 3).astype(np.float32)
    input_dict_1 = {
        "input": input_1,
        "depth_radius": 5,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "LRN_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = np.random.rand(1, 3, 3, 2).astype(np.float32)
    input_dict_2 = {
        "input": input_2,
        "depth_radius": 3,
        "bias": 2.0,
        "alpha": 0.5,
        "beta": 1.0,
        "name": "LRN_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_3 = np.random.rand(4, 2, 2, 1).astype(np.float32)
    input_dict_3 = {
        "input": input_3,
        "depth_radius": 1,
        "bias": 0.5,
        "alpha": 2.0,
        "beta": 0.2,
        "name": "LRN_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_4 = np.random.rand(2, 2, 2, 4).astype(np.float32)
    input_dict_4 = {
        "input": input_4,
        "depth_radius": 2,
        "bias": 1.5,
        "alpha": 0.8,
        "beta": 0.7,
        "name": "LRN_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_5 = np.random.rand(3, 3, 3, 5).astype(np.float32)
    input_dict_5 = {
        "input": input_5,
        "depth_radius": 4,
        "bias": 0.1,
        "alpha": 1.2,
        "beta": 0.3,
        "name": "LRN_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    input_6 = np.random.rand(1, 1, 1, 1).astype(np.float32)
    input_dict_6 = {
        "input": input_6,
        "depth_radius": 0,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "LRN_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    input_7 = np.random.rand(2, 2, 2, 2).astype(np.float32)
    input_dict_7 = {
        "input": input_7,
        "depth_radius": 1,
        "bias": -1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "LRN_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    input_8 = np.random.rand(4, 4, 4, 3).astype(np.float32)
    input_dict_8 = {
        "input": input_8,
        "depth_radius": 5,
        "bias": -0.5,
        "alpha": -0.2,
        "beta": 1.0,
        "name": "LRN_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    input_9 = np.random.rand(1, 2, 2, 3).astype(np.float32)
    input_dict_9 = {
        "input": input_9,
        "depth_radius": 1,
        "bias": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "name": "LRN_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    input_10 = np.random.rand(2, 3, 3, 4).astype(np.float32)
    input_dict_10 = {
        "input": input_10,
        "depth_radius": 2,
        "bias": 3.0,
        "alpha": 1.5,
        "beta": 0.8,
        "name": "LRN_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.LRN"] = tf_raw_ops_LRN_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.LRN' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LRN'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.LRN', generated_inputs['tf.raw_ops.LRN'], lib="tf", suffix=0)
