
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_boolean_mask_inputs():
    list_of_inputs = []

    # Input 1
    tensor = np.array([0, 1, 2, 3])
    mask = np.array([True, False, True, False])
    axis = np.array(0)
    name = "bool_mask_1"
    input_dict = {"tensor": tensor, "mask": mask, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tensor = np.array([[1, 2], [3, 4], [5, 6]])
    mask = np.array([True, False, True])
    axis = np.array(0)
    name = "bool_mask_2"
    input_dict = {"tensor": tensor, "mask": mask, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    mask = np.array([True, False])
    axis = np.array(0)
    name = "bool_mask_3"
    input_dict = {"tensor": tensor, "mask": mask, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mask = np.array([[True, False, True], [False, True, False], [True, True, True]])
    axis = np.array(0)
    name = "bool_mask_4"
    input_dict = {"tensor": tensor, "mask": mask, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    mask = np.array([[True, False], [False, True]])
    axis = np.array(1)
    name = "bool_mask_5"
    input_dict = {"tensor": tensor, "mask": mask, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    tensor = np.array([1, 2, 3, 4, 5, 6])
    mask = np.array([False, False, True, True, False, True])
    axis = np.array(0)
    name = "bool_mask_6"
    input_dict = {"tensor": tensor, "mask": mask, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    tensor = np.array([[1, 2, 3], [4, 5, 6]])
    mask = np.array([True, False])
    axis = np.array(0)
    name = "bool_mask_7"
    input_dict = {"tensor": tensor, "mask": mask, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    mask = np.array([[[True, False], [False, True]], [[False, True], [True, False]]])
    axis = np.array(0)
    name = "bool_mask_8"
    input_dict = {"tensor": tensor, "mask": mask, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    tensor = np.array([[1, 2], [3, 4], [5, 6]])
    mask = np.array([True, False, True])
    axis = np.array(0)
    name = "bool_mask_9"
    input_dict = {"tensor": tensor, "mask": mask, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    tensor = np.array([1, 2, 3, 4, 5])
    mask = np.array([True, True, False, False, True])
    axis = np.array(0)
    name = "bool_mask_10"
    input_dict = {"tensor": tensor, "mask": mask, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.boolean_mask"] = tf_boolean_mask_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.boolean_mask' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.boolean_mask'.")

check_valid('tf.boolean_mask', generated_inputs['tf.boolean_mask'], lib="tf", suffix=0)
