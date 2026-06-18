
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_strings_as_string_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        "input": np.array([1, 2, 3], dtype=np.int32),
        "precision": -1,
        "scientific": False,
        "shortest": False,
        "width": -1,
        "fill": "",
        "name": "as_string_1"
    })

    # Input 2
    list_of_inputs.append({
        "input": np.array([3.14159, 2.71828], dtype=np.float32),
        "precision": 2,
        "scientific": False,
        "shortest": False,
        "width": -1,
        "fill": "",
        "name": "as_string_2"
    })

    # Input 3
    list_of_inputs.append({
        "input": np.array([-10, 0, 100], dtype=np.int64),
        "precision": -1,
        "scientific": False,
        "shortest": False,
        "width": 4,
        "fill": "0",
        "name": "as_string_3"
    })

    # Input 4
    list_of_inputs.append({
        "input": np.array([123.456, 789.012], dtype=np.float64),
        "precision": 4,
        "scientific": True,
        "shortest": False,
        "width": 10,
        "fill": " ",
        "name": "as_string_4"
    })

    # Input 5
    list_of_inputs.append({
        "input": np.array([True, False, True], dtype=np.bool_),
        "precision": -1,
        "scientific": False,
        "shortest": False,
        "width": -1,
        "fill": "",
        "name": "as_string_5"
    })

    # Input 6
    list_of_inputs.append({
        "input": np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32),
        "precision": 1,
        "scientific": False,
        "shortest": True,
        "width": -1,
        "fill": "",
        "name": "as_string_6"
    })

    # Input 7
    list_of_inputs.append({
        "input": np.array([10000, 20000], dtype=np.int32),
        "precision": -1,
        "scientific": False,
        "shortest": False,
        "width": 6,
        "fill": "0",
        "name": "as_string_7"
    })

    # Input 8
    list_of_inputs.append({
        "input": np.array([1e-10, 1e10], dtype=np.float64),
        "precision": 3,
        "scientific": True,
        "shortest": False,
        "width": 12,
        "fill": " ",
        "name": "as_string_8"
    })

    # Input 9
    list_of_inputs.append({
        "input": np.array([-5, 5], dtype=np.int32),
        "precision": -1,
        "scientific": False,
        "shortest": False,
        "width": 3,
        "fill": "0",
        "name": "as_string_9"
    })

    # Input 10
    list_of_inputs.append({
        "input": np.array([1.1, 2.2, 3.3], dtype=np.float32),
        "precision": 0,
        "scientific": False,
        "shortest": False,
        "width": -1,
        "fill": "",
        "name": "as_string_10"
    })

    return list_of_inputs

generated_inputs["tf.strings.as_string"] = tf_strings_as_string_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.strings.as_string' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.as_string'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.strings.as_string', generated_inputs['tf.strings.as_string'], lib="tf", suffix=0)
