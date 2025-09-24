
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_log_softmax_inputs():
    list_of_inputs = []

    # Input 1
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis = None
    name = "log_softmax_1"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    logits = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    axis = 1
    name = "log_softmax_2"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    logits = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    axis = 0
    name = "log_softmax_3"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    logits = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    axis = None
    name = "log_softmax_4"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    logits = np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float64)
    axis = 1
    name = "log_softmax_5"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    logits = np.array([[[1.0, 2.0], [-3.0, -4.0]], [[-5.0, 6.0], [7.0, -8.0]]], dtype=np.float32)
    axis = 2
    name = "log_softmax_6"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    logits = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    axis = None
    name = "log_softmax_7"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    logits = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    axis = 1
    name = "log_softmax_8"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = 0
    name = "log_softmax_9"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    logits = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float64)
    axis = 0
    name = "log_softmax_10"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.log_softmax"] = tf_nn_log_softmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.log_softmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.log_softmax'.")

check_valid('tf.nn.log_softmax', generated_inputs['tf.nn.log_softmax'], lib="tf", suffix=0)
