
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_PlaceholderWithDefault_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "name": "placeholder_1",
        "input": np.array([1, 2, 3], dtype=np.int32),
        "shape": [3]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "name": "placeholder_2",
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "shape": [2, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "name": "placeholder_3",
        "input": np.array(42, dtype=np.int64),
        "shape": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "name": "placeholder_4",
        "input": np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]], dtype=np.float64),
        "shape": [2, 2, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "name": "placeholder_5",
        "input": np.array([True, False, True], dtype=np.bool_),
        "shape": [3]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "name": "placeholder_6",
        "input": np.array([[-1, -2], [-3, -4]], dtype=np.int16),
        "shape": [2, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "name": "placeholder_7",
        "input": np.ones((1, 2, 2, 1), dtype=np.uint8),
        "shape": [1, 2, 2, 1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "name": "placeholder_8",
        "input": np.array([0.1, -0.2, 0.5], dtype=np.float16),
        "shape": [3]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "name": "placeholder_9",
        "input": np.zeros((2, 2, 2), dtype=np.int32),
        "shape": [2, 2, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "name": "placeholder_10",
        "input": np.ones((1, 1, 1, 1, 1), dtype=np.float32),
        "shape": [1, 1, 1, 1, 1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.PlaceholderWithDefault"] = tf_raw_ops_PlaceholderWithDefault_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.PlaceholderWithDefault' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.PlaceholderWithDefault'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.PlaceholderWithDefault', generated_inputs['tf.raw_ops.PlaceholderWithDefault'], lib="tf", suffix=0)
