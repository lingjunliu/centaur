
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_mask_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([0, 2])
    values = np.array([[1, 2], [3, 4]])
    dense_shape = [10, 2]
    a = tf.IndexedSlices(values=tf.constant(values), indices=tf.constant(indices), dense_shape=tf.constant(dense_shape))
    mask_indices = tf.constant([0])
    name = "mask1"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([1, 3])
    values = np.array([[9, 10], [11, 12]])
    dense_shape = [10, 2]
    a = tf.IndexedSlices(values=tf.constant(values), indices=tf.constant(indices), dense_shape=tf.constant(dense_shape))
    mask_indices = tf.constant([1])
    name = "mask2"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([0, 1])
    values = np.array([[1, 2, 3], [4, 5, 6]])
    dense_shape = [5, 3]
    a = tf.IndexedSlices(values=tf.constant(values), indices=tf.constant(indices), dense_shape=tf.constant(dense_shape))
    mask_indices = tf.constant([0])
    name = "mask3"
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
