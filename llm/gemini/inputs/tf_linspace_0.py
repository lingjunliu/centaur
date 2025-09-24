
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linspace_inputs():
    list_of_inputs = []

    # Input 1
    start = np.float32(0.0)
    stop = np.float32(10.0)
    num = 5
    name = "linspace_1"
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "name": name, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    start = np.float32(1.0)
    stop = np.float32(5.0)
    num = 10
    name = "linspace_2"
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "name": name, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    start = np.float32(-5.0)
    stop = np.float32(5.0)
    num = 20
    name = "linspace_3"
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "name": name, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    start = np.array([0.0, 1.0], dtype=np.float32)
    stop = np.array([10.0, 11.0], dtype=np.float32)
    num = 5
    name = "linspace_4"
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "name": name, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    start = np.array([0.0, 1.0], dtype=np.float32)
    stop = np.array([10.0, 11.0], dtype=np.float32)
    num = 5
    name = "linspace_5"
    axis = -1
    input_dict = {"start": start, "stop": stop, "num": num, "name": name, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    start = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    stop = np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32)
    num = 5
    name = "linspace_6"
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "name": name, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    start = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    stop = np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32)
    num = 5
    name = "linspace_7"
    axis = 1
    input_dict = {"start": start, "stop": stop, "num": num, "name": name, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    start = np.array([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]], dtype=np.float32)
    stop = np.array([[[10.0, 11.0], [12.0, 13.0]], [[14.0, 15.0], [16.0, 17.0]]], dtype=np.float32)
    num = 5
    name = "linspace_8"
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "name": name, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    start = np.array([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]], dtype=np.float32)
    stop = np.array([[[10.0, 11.0], [12.0, 13.0]], [[14.0, 15.0], [16.0, 17.0]]], dtype=np.float32)
    num = 5
    name = "linspace_9"
    axis = 1
    input_dict = {"start": start, "stop": stop, "num": num, "name": name, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    start = np.array([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]], dtype=np.float32)
    stop = np.array([[[10.0, 11.0], [12.0, 13.0]], [[14.0, 15.0], [16.0, 17.0]]], dtype=np.float32)
    num = 5
    name = "linspace_10"
    axis = 2
    input_dict = {"start": start, "stop": stop, "num": num, "name": name, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    start = np.float64(0.0)
    stop = np.float64(10.0)
    num = 5
    name = "linspace_11"
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "name": name, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    start = np.array([0.0, 1.0], dtype=np.float64)
    stop = np.array([10.0, 11.0], dtype=np.float64)
    num = 5
    name = "linspace_12"
    axis = 0
    input_dict = {"start": start, "stop": stop, "num": num, "name": name, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linspace"] = tf_linspace_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linspace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linspace'.")

check_valid('tf.linspace', generated_inputs['tf.linspace'], lib="tf", suffix=0)
