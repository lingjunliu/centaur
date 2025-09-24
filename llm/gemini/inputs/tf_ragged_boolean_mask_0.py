
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_boolean_mask_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D example
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mask = np.array([[True, False, True], [False, True, False], [True, True, True]])
    name = "mask1"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: RaggedTensor data, RaggedTensor mask
    data = tf.ragged.constant([[1, 2, 3], [4], [5, 6]]).to_tensor().numpy()
    mask = tf.ragged.constant([[False, False, True], [True], [True, False]]).to_tensor().numpy()
    name = "mask2"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Masking rows
    data = tf.ragged.constant([[1, 2, 3], [4], [5, 6]]).to_tensor().numpy()
    mask = np.array([True, False, True])
    name = "mask3"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D data and 2D mask
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    mask = np.array([[True, False], [False, True]])
    name = "mask4"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D data and 3D mask
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    mask = np.array([[[True, False], [False, True]], [[False, True], [True, False]]])
    name = "mask5"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Ragged Tensor data, Dense Mask
    data = tf.ragged.constant([[1, 2, 3], [4], [5, 6]]).to_tensor().numpy()
    mask = np.array([[True, False, True], [True, False, False], [False, True, True]])
    name = "mask6"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D data and 1D mask
    data = np.array([1, 2, 3, 4, 5])
    mask = np.array([True, False, True, False, True])
    name = "mask7"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Ragged rank 1, Ragged mask rank 1
    data = tf.ragged.constant([[1, 2], [], [3, 4, 5]]).to_tensor().numpy()
    mask = tf.ragged.constant([[True, False], [False], [False, True, True]]).to_tensor().numpy()
    name = "mask8"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Ragged rank 0, Dense mask rank 0
    data = np.array([1, 2, 3])
    mask = np.array([True, False, True])
    name = "mask9"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.ragged.boolean_mask"] = tf_ragged_boolean_mask_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.ragged.boolean_mask' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.boolean_mask'.")

check_valid('tf.ragged.boolean_mask', generated_inputs['tf.ragged.boolean_mask'], lib="tf", suffix=0)
