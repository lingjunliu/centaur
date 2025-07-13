
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_betainc_inputs():
    list_of_inputs = []

    # Input 1, valid
    a = np.array(0.5, dtype=np.float32)
    b = np.array(0.5, dtype=np.float32)
    x = np.array(0.5, dtype=np.float32)
    name = "betainc_1"

    input_dict = {
        "a": a,
        "b": b,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    a = np.array(2.0, dtype=np.float64)
    b = np.array(3.0, dtype=np.float64)
    x = np.array(0.8, dtype=np.float64)
    name = "betainc_2"

    input_dict = {
        "a": a,
        "b": b,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid, array inputs
    a = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    b = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    x = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    name = "betainc_3"

    input_dict = {
        "a": a,
        "b": b,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid, larger array
    a = np.random.rand(2, 3).astype(np.float64)
    b = np.random.rand(2, 3).astype(np.float64)
    x = np.random.rand(2, 3).astype(np.float64)
    name = "betainc_4"

    input_dict = {
        "a": a,
        "b": b,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, different values
    a = np.array(0.1, dtype=np.float32)
    b = np.array(0.9, dtype=np.float32)
    x = np.array(0.3, dtype=np.float32)
    name = "betainc_5"

    input_dict = {
        "a": a,
        "b": b,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    a = np.array(0.75, dtype=np.float64)
    b = np.array(0.25, dtype=np.float64)
    x = np.array(0.6, dtype=np.float64)
    name = "betainc_6"

    input_dict = {
        "a": a,
        "b": b,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, x close to 1
    a = np.array(0.6, dtype=np.float32)
    b = np.array(0.4, dtype=np.float32)
    x = np.array(0.99, dtype=np.float32)
    name = "betainc_7"

    input_dict = {
        "a": a,
        "b": b,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid, x close to 0
    a = np.array(0.8, dtype=np.float64)
    b = np.array(0.2, dtype=np.float64)
    x = np.array(0.01, dtype=np.float64)
    name = "betainc_8"

    input_dict = {
        "a": a,
        "b": b,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid, different shape arrays
    a = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    b = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    x = np.array([[0.9, 0.10], [0.11, 0.12]], dtype=np.float32)
    name = "betainc_9"

    input_dict = {
        "a": a,
        "b": b,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10, valid, 3D array
    a = np.random.rand(2, 2, 2).astype(np.float64)
    b = np.random.rand(2, 2, 2).astype(np.float64)
    x = np.random.rand(2, 2, 2).astype(np.float64)
    name = "betainc_10"

    input_dict = {
        "a": a,
        "b": b,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Betainc"] = tf_raw_ops_betainc_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Betainc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Betainc'.")

check_valid('tf.raw_ops.Betainc', generated_inputs['tf.raw_ops.Betainc'], lib="tf", suffix=0)
