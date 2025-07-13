
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_next_after_inputs():
    list_of_inputs = []

    # Input 1: Basic float32
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([1.1, 1.9, 3.1], dtype=np.float32)
    name = "next_after_1"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic float64
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x2 = np.array([1.1, 1.9, 3.1], dtype=np.float64)
    name = "next_after_2"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values, float32
    x1 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    x2 = np.array([-0.9, -2.1, -2.9], dtype=np.float32)
    name = "next_after_3"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values, float64
    x1 = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    x2 = np.array([-0.9, -2.1, -2.9], dtype=np.float64)
    name = "next_after_4"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero values, float32
    x1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    x2 = np.array([0.1, -0.1, 0.0], dtype=np.float32)
    name = "next_after_5"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zero values, float64
    x1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    x2 = np.array([0.1, -0.1, 0.0], dtype=np.float64)
    name = "next_after_6"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, float32
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x2 = np.array([[1.1, 1.9], [3.1, 3.9]], dtype=np.float32)
    name = "next_after_7"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, float64
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    x2 = np.array([[1.1, 1.9], [3.1, 3.9]], dtype=np.float64)
    name = "next_after_8"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different directions, float32
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([0.9, 2.1, 2.9], dtype=np.float32)
    name = "next_after_9"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different directions, float64
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x2 = np.array([0.9, 2.1, 2.9], dtype=np.float64)
    name = "next_after_10"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_next_after_inputs()
generated_inputs["tf.raw_ops.NextAfter"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.NextAfter"].append({
        "x1": tf.constant(input_dict["x1"].tolist()),
        "x2": tf.constant(input_dict["x2"].tolist()),
        "name": input_dict["name"]
    })

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.NextAfter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NextAfter'.")

check_valid('tf.raw_ops.NextAfter', generated_inputs['tf.raw_ops.NextAfter'], lib="tf", suffix=0)
