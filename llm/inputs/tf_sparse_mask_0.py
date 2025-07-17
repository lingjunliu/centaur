
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_mask_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([0, 2, 4, 6])
    values = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    dense_shape = [10, 2]
    a = tf.IndexedSlices(values=tf.constant(values), indices=tf.constant(indices), dense_shape=tf.constant(dense_shape))
    mask_indices = tf.constant([0, 6])
    name = "mask1"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([1, 3, 5, 7, 9])
    values = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    dense_shape = [12]
    a = tf.IndexedSlices(values=tf.constant(values), indices=tf.constant(indices), dense_shape=tf.constant(dense_shape))
    mask_indices = tf.constant([1, 5, 9])
    name = "mask2"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([0, 1, 2])
    values = np.array([["a", "b"], ["c", "d"], ["e", "f"]])
    dense_shape = [5, 2]
    a = tf.IndexedSlices(values=tf.constant(values), indices=tf.constant(indices), dense_shape=tf.constant(dense_shape))
    mask_indices = tf.constant([0])
    name = "mask3"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = np.array([2, 4, 6, 8])
    values = np.array([10, 20, 30, 40])
    dense_shape = [10]
    a = tf.IndexedSlices(values=tf.constant(values), indices=tf.constant(indices), dense_shape=tf.constant(dense_shape))
    mask_indices = tf.constant([2, 8])
    name = "mask4"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices = np.array([10, 20, 30])
    values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dense_shape = [40, 3]
    a = tf.IndexedSlices(values=tf.constant(values), indices=tf.constant(indices), dense_shape=tf.constant(dense_shape))
    mask_indices = tf.constant([20])
    name = "mask5"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices = np.array([0, 1, 2, 3, 4])
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    dense_shape = [5]
    a = tf.IndexedSlices(values=tf.constant(values), indices=tf.constant(indices), dense_shape=tf.constant(dense_shape))
    mask_indices = tf.constant([0, 1, 2, 3, 4])
    name = "mask6"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    indices = np.array([1, 3, 5])
    values = np.array([[-1, -2], [-3, -4], [-5, -6]])
    dense_shape = [7, 2]
    a = tf.IndexedSlices(values=tf.constant(values), indices=tf.constant(indices), dense_shape=tf.constant(dense_shape))
    mask_indices = tf.constant([1, 3, 5])
    name = "mask7"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    indices = np.array([1, 3, 5])
    values = np.array([[-1, -2], [-3, -4], [-5, -6]])
    dense_shape = [7, 2]
    a = tf.IndexedSlices(values=tf.constant(values), indices=tf.constant(indices), dense_shape=tf.constant(dense_shape))
    mask_indices = tf.constant([])
    name = "mask8"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    indices = np.array([0])
    values = np.array([[1, 2, 3, 4]])
    dense_shape = [10, 4]
    a = tf.IndexedSlices(values=tf.constant(values), indices=tf.constant(indices), dense_shape=tf.constant(dense_shape))
    mask_indices = tf.constant([0])
    name = "mask9"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    indices = np.array([2, 5, 8])
    values = np.array([100, 200, 300])
    dense_shape = [10]
    a = tf.IndexedSlices(values=tf.constant(values), indices=tf.constant(indices), dense_shape=tf.constant(dense_shape))
    mask_indices = tf.constant([5])
    name = "mask10"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.mask"] = tf_sparse_mask_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.mask' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.mask'.")

check_valid('tf.sparse.mask', generated_inputs['tf.sparse.mask'], lib="tf", suffix=0)
