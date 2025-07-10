
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_boolean_mask_inputs():
    list_of_inputs = []

    # Input 1: 2D Tensor, 2D Mask
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mask = np.array([[True, False, True], [False, True, False], [True, True, False]])
    name = "mask1"
    input_dict = {"data": tf.constant(data), "mask": tf.constant(mask, dtype=tf.bool), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D RaggedTensor, 2D Ragged Mask
    data = tf.ragged.constant([[1, 2, 3], [4], [5, 6]])
    mask = tf.ragged.constant([[False, False, True], [True], [True, False]])
    name = "mask2"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D RaggedTensor, 1D Mask
    data = tf.ragged.constant([[1, 2, 3], [4], [5, 6]])
    mask = np.array([True, False, True])
    name = "mask3"
    input_dict = {"data": data, "mask": tf.constant(mask, dtype=tf.bool), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D Tensor, 2D Mask
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    mask = np.array([[True, False], [False, True]])
    name = "mask4"
    input_dict = {"data": tf.constant(data), "mask": tf.constant(mask, dtype=tf.bool), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D RaggedTensor, 2D Ragged Mask
    data = tf.ragged.constant([[[1, 2], [3]], [[4, 5, 6], [7]]])
    mask = tf.ragged.constant([[True, False], [False, True]])
    name = "mask5"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D Tensor, 3D Mask (prefix shape)
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    mask = np.array([[[True, False], [False, True]], [[True, True], [False, False]]])
    name = "mask6"
    input_dict = {"data": tf.constant(data), "mask": tf.constant(mask, dtype=tf.bool), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D Tensor, 1D Mask
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mask = np.array([True, False, True])
    name = "mask7"
    input_dict = {"data": tf.constant(data), "mask": tf.constant(mask, dtype=tf.bool), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D Tensor, 1D Mask
    data = np.array([1, 2, 3, 4, 5])
    mask = np.array([True, False, True, False, True])
    name = "mask8"
    input_dict = {"data": tf.constant(data), "mask": tf.constant(mask, dtype=tf.bool), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D RaggedTensor, 2D Tensor Mask
    data = tf.ragged.constant([[1, 2, 3], [4], [5, 6]])
    mask = np.array([[False, False, True], [True, False, False], [True, False, True]], dtype=bool)
    name = "mask9"
    input_dict = {"data": data, "mask": tf.constant(mask), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10:  3D RaggedTensor, 2D Mask
    data = tf.ragged.constant([[[1, 2], [3]], [[4, 5, 6], [7]]])
    mask = np.array([[True, False], [False, True]], dtype=bool)
    name = "mask10"
    input_dict = {"data": data, "mask": tf.constant(mask), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D Ragged Tensor, 1D Mask.
    data = tf.ragged.constant([1,2,3,4])
    mask = np.array([True, False, True, False], dtype=bool)
    name = "mask11"
    input_dict = {"data": data, "mask": tf.constant(mask), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Empty Ragged Tensor, Empty Mask.
    data = tf.ragged.constant([[]])
    mask = tf.ragged.constant([[False]])
    name = "mask12"
    input_dict = {"data": data, "mask": mask, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Scalar data, Scalar mask
    data = np.array(1)
    mask = np.array(True)
    name = "mask13"
    input_dict = {"data": tf.constant(data), "mask": tf.constant(mask), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14: 1D tensor, 0D mask
    data = np.array([1, 2, 3])
    mask = np.array(True)
    name = "mask14"
    input_dict = {"data": tf.constant(data), "mask": tf.constant(mask), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 15: 0D tensor, 0D mask
    data = np.array(5)
    mask = np.array(False)
    name = "mask15"
    input_dict = {"data": tf.constant(data), "mask": tf.constant(mask), "name": name}
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
