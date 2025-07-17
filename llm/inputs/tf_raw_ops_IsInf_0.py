
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_isinf_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([np.inf, -np.inf, 1.0, 0.0, -1.0], dtype=np.float32)
    input_dict = {"x": x, "name": "is_inf_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[np.inf, 1.0], [-np.inf, 0.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "is_inf_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[[np.inf, -np.inf], [1.0, 0.0]], [[-1.0, np.inf], [0.0, -np.inf]]], dtype=np.float32)
    input_dict = {"x": x, "name": "is_inf_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([np.inf, -np.inf, 1.0, 0.0, -1.0], dtype=np.float64)
    input_dict = {"x": x, "name": "is_inf_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    input_dict = {"x": x, "name": "is_inf_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    input_dict = {"x": x, "name": "is_inf_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    input_dict = {"x": x, "name": "is_inf_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([np.inf], dtype=np.float32)
    input_dict = {"x": x, "name": "is_inf_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([-np.inf], dtype=np.float64)
    input_dict = {"x": x, "name": "is_inf_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    x = np.array([[np.inf, -np.inf], [1.0, 0.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "is_inf_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.IsInf"] = tf_raw_ops_isinf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.IsInf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.IsInf'.")

check_valid('tf.raw_ops.IsInf', generated_inputs['tf.raw_ops.IsInf'], lib="tf", suffix=0)
