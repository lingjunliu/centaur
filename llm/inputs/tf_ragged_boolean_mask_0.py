
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_boolean_mask_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D Tensor mask
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mask = np.array([[True, False, True], [False, False, False], [True, False, False]])
    name = "basic_mask"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D RaggedTensor mask
    data = np.array([[1, 2, 3], [4], [5, 6]], dtype=object)
    mask = np.array([[False, False, True], [False], [True, True]], dtype=object)
    name = "ragged_mask"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Mask rows of a 2D RaggedTensor
    data = np.array([[1, 2, 3], [4], [5, 6]], dtype=object)
    mask = np.array([True, False, True])
    name = "row_mask"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D Tensor mask (prefix match)
    data = np.arange(1, 28).reshape((3, 3, 3))
    mask = np.array([[True, False, True], [False, True, False], [True, False, True]])
    name = "3d_mask"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D data and 1D mask
    data = np.array([1, 2, 3, 4, 5])
    mask = np.array([True, False, True, True, False])
    name = "1d_mask"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: Ragged 3D
    data = np.array([[[1,2],[3]], [[4,5,6],[7,8]]], dtype=object)
    mask = np.array([[[True,False],[True]], [[False,True,True],[False,False]]], dtype=object)
    name = "ragged_3d"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  Ragged 3D mask prefix
    data = np.array([[[1,2],[3]], [[4,5,6],[7,8]]], dtype=object)
    mask = np.array([[True, True], [False, True]])
    name = "ragged_3d_prefix"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty data
    data = np.array([])
    mask = np.array([])
    name = "empty"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Ragged Data with empty rows
    data = np.array([[], [1,2], [3]], dtype=object)
    mask = np.array([[], [True, False], [True]], dtype=object)
    name = "ragged_empty_rows"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Ragged Data with scalar values.
    data = np.array([[1], [2, 3], [4]], dtype=object)
    mask = np.array([[True], [False, True], [True]], dtype=object)
    name = "ragged_scalar_values"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: All false mask
    data = np.array([[1, 2, 3], [4, 5, 6]])
    mask = np.array([[False, False, False], [False, False, False]])
    name = "all_false"
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
