
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_squeeze_inputs():
    list_of_inputs = []

    # Input 1
    input_val = np.random.randn(1, 2, 1).astype(np.float32)
    axis = [0]
    name = "squeeze_1"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    # Input 2
    input_val = np.random.randint(0, 10, size=(1, 1, 3, 1)).astype(np.int32)
    axis = [0, 1]
    name = "squeeze_2"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    # Input 3
    input_val = np.random.choice([True, False], size=(1, 5)).astype(np.bool_)
    axis = [0]
    name = "squeeze_3"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    # Input 4
    input_val = np.random.randn(2, 1, 3, 1, 4).astype(np.float64)
    axis = [1, 3]
    name = "squeeze_4"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    # Input 5
    input_val = np.random.randint(-5, 5, size=(1,)).astype(np.int64)
    axis = [0]
    name = "squeeze_5"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    # Input 6
    input_val = np.random.randn(1, 2, 1).astype(np.float32)
    axis = [-1]
    name = "squeeze_6"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    # Input 7
    input_val = np.random.randn(1, 1, 1, 1, 1, 1).astype(np.float32)
    axis = [0, 1, 2, 3, 4, 5]
    name = "squeeze_7"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    # Input 8
    real = np.random.randn(1, 4, 1, 4).astype(np.float32)
    imag = np.random.randn(1, 4, 1, 4).astype(np.float32)
    input_val = (real + 1j * imag).astype(np.complex64)
    axis = [0, 2]
    name = "squeeze_8"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    # Input 9
    input_val = np.random.randint(0, 255, size=(3, 1, 2)).astype(np.uint8)
    axis = [1]
    name = "squeeze_9"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    # Input 10
    input_val = np.random.randn(1, 2, 1, 1, 3).astype(np.float32)
    axis = [-3, -2]
    name = "squeeze_10"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.squeeze"] = tf_squeeze_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.squeeze' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.squeeze'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.squeeze', generated_inputs['tf.squeeze'], lib="tf", suffix=0)
