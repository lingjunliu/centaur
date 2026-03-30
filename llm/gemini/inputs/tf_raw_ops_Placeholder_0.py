
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_placeholder_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "dtype": tf.float32,
        "shape": [],
        "name": "placeholder_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "dtype": tf.int32,
        "shape": [2, 3],
        "name": "placeholder_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "dtype": tf.bool,
        "shape": [5],
        "name": "placeholder_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "dtype": tf.string,
        "shape": [1, 4, 2],
        "name": "placeholder_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "dtype": tf.uint8,
        "shape": [10, 10, 3],
        "name": "placeholder_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "dtype": tf.int64,
        "shape": [1],
        "name": "placeholder_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "dtype": tf.complex64,
        "shape": [2, 2, 2, 2],
        "name": "placeholder_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "dtype": tf.float64,
        "shape": [],
        "name": "placeholder_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "dtype": tf.bfloat16,
        "shape": [128],
        "name": "placeholder_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "dtype": tf.qint8,
        "shape": [3, 5],
        "name": "placeholder_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Placeholder"] = tf_raw_ops_placeholder_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Placeholder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Placeholder'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Placeholder', generated_inputs['tf.raw_ops.Placeholder'], lib="tf", suffix=0)
