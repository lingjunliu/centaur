
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_extract_volume_patches_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "input": np.random.randn(1, 3, 3, 3, 1).astype(np.float32),
        "ksizes": [1, 2, 2, 2, 1],
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "name": "extract_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "input": np.random.randint(-10, 10, size=(2, 4, 4, 4, 3)).astype(np.int32),
        "ksizes": [1, 3, 3, 3, 1],
        "strides": [1, 2, 2, 2, 1],
        "padding": "SAME",
        "name": "extract_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "input": np.random.randint(0, 255, size=(1, 5, 5, 5, 2)).astype(np.uint8),
        "ksizes": [1, 1, 1, 1, 1],
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "name": "extract_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "input": np.random.randn(3, 2, 2, 2, 4).astype(np.float64),
        "ksizes": [1, 2, 2, 2, 1],
        "strides": [1, 1, 1, 1, 1],
        "padding": "SAME",
        "name": "extract_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "input": np.random.randn(1, 10, 10, 10, 1).astype(np.float32),
        "ksizes": [1, 5, 5, 5, 1],
        "strides": [1, 5, 5, 5, 1],
        "padding": "VALID",
        "name": "extract_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "input": np.random.randint(-1000, 1000, size=(2, 6, 6, 6, 2)).astype(np.int64),
        "ksizes": [1, 2, 3, 2, 1],
        "strides": [1, 1, 2, 1, 1],
        "padding": "SAME",
        "name": "extract_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "input": np.random.randn(1, 3, 4, 5, 1).astype(np.float32),
        "ksizes": [1, 2, 2, 2, 1],
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "name": "extract_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "input": np.random.randint(-50, 50, size=(4, 3, 3, 3, 2)).astype(np.int32),
        "ksizes": [1, 3, 2, 3, 1],
        "strides": [1, 1, 1, 1, 1],
        "padding": "SAME",
        "name": "extract_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "input": np.random.randint(0, 255, size=(1, 8, 8, 8, 3)).astype(np.uint8),
        "ksizes": [1, 4, 4, 4, 1],
        "strides": [1, 2, 2, 2, 1],
        "padding": "VALID",
        "name": "extract_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "input": np.random.randn(2, 2, 3, 4, 2).astype(np.float64),
        "ksizes": [1, 1, 2, 3, 1],
        "strides": [1, 1, 1, 1, 1],
        "padding": "SAME",
        "name": "extract_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.extract_volume_patches"] = tf_extract_volume_patches_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.extract_volume_patches' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.extract_volume_patches'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.extract_volume_patches', generated_inputs['tf.extract_volume_patches'], lib="tf", suffix=0)
