
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_where_inputs():
    list_of_inputs = []

    # Input 1: bool condition, 1D
    condition = np.array([True, False, True, False])
    name = "bool_1d"
    input_dict = {"condition": condition, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: bool condition, 2D
    condition = np.array([[True, False], [False, True]])
    name = "bool_2d"
    input_dict = {"condition": condition, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: numeric condition (int), 1D
    condition = np.array([1, 0, 2, -1])
    name = "int_1d"
    input_dict = {"condition": condition, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: numeric condition (float), 2D
    condition = np.array([[0.1, 0.0], [-0.5, 2.0]])
    name = "float_2d"
    input_dict = {"condition": condition, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: bool condition, 3D
    condition = np.array([[[True, False], [True, True]], [[False, False], [True, False]]])
    name = "bool_3d"
    input_dict = {"condition": condition, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: numeric condition (int), 3D
    condition = np.array([[[1, 0], [2, 3]], [[0, -1], [4, 0]]])
    name = "int_3d"
    input_dict = {"condition": condition, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: numeric condition (float), 1D with negative values and zeros
    condition = np.array([-1.0, 0.0, 2.5, -0.0])
    name = "float_1d_neg"
    input_dict = {"condition": condition, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: bool condition with all False values
    condition = np.array([False, False, False])
    name = "bool_all_false"
    input_dict = {"condition": condition, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: bool condition with all True values
    condition = np.array([True, True, True])
    name = "bool_all_true"
    input_dict = {"condition": condition, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.where_1"] = tf_where_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.where_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.where_1'.")

check_valid('tf.where', generated_inputs['tf.where_1'], lib="tf", suffix=1)
