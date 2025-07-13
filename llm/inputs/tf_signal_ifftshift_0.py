
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_ifftshift_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([0, 1, 2, 3])
    axes = (0,)
    name = "ifftshift_example_1"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    axes = (0, 1)
    name = "ifftshift_example_2"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    axes = (0, 1, 2)
    name = "ifftshift_example_3"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([1, 2, 3, 4, 5])
    axes = (0,)
    name = "ifftshift_example_4"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([[1, 2, 3], [4, 5, 6]])
    axes = (1,)
    name = "ifftshift_example_5"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([1, 2, 3, 4, 5, 6])
    axes = (0,)
    name = "ifftshift_example_6"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([[1, 2], [3, 4]])
    axes = (0, 1)
    name = "ifftshift_example_7"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    x = np.array([1, 2, 3])
    axes = (0,)
    name = "ifftshift_example_8"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([[1, 2], [3, 4], [5, 6]])
    axes = (0,)
    name = "ifftshift_example_9"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    axes = (0, 1)
    name = "ifftshift_example_10"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.ifftshift"] = tf_signal_ifftshift_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.ifftshift' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.ifftshift'.")

check_valid('tf.signal.ifftshift', generated_inputs['tf.signal.ifftshift'], lib="tf", suffix=0)
