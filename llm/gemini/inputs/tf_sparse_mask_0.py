
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_mask_inputs():
    list_of_inputs = []

    # Input 1
    indices = np.array([0, 2, 4, 6])
    values = np.array([1, 3, 5, 7])
    dense_shape = tf.constant([10])
    a = tf.IndexedSlices(values=values, indices=indices, dense_shape=dense_shape)
    mask_indices = np.array([0, 4])
    name = "mask_op_1"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([1, 3, 5, 7, 9])
    values = np.array([2, 4, 6, 8, 10])
    dense_shape = tf.constant([12])
    a = tf.IndexedSlices(values=values, indices=indices, dense_shape=dense_shape)
    mask_indices = np.array([3, 9])
    name = "mask_op_2"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([0, 1, 2])
    values = np.array([[1, 2], [3, 4], [5, 6]])
    dense_shape = tf.constant([5, 2])
    a = tf.IndexedSlices(values=values, indices=indices, dense_shape=dense_shape)
    mask_indices = np.array([0, 2])
    name = "mask_op_3"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = np.array([2, 5, 8])
    values = np.array([[7, 8], [9, 10], [11, 12]])
    dense_shape = tf.constant([10, 2])
    a = tf.IndexedSlices(values=values, indices=indices, dense_shape=dense_shape)
    mask_indices = np.array([5])
    name = "mask_op_4"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices = np.array([0, 1, 2, 3, 4])
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    dense_shape = tf.constant([6])
    a = tf.IndexedSlices(values=values, indices=indices, dense_shape=dense_shape)
    mask_indices = np.array([1, 3, 4])
    name = "mask_op_5"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    indices = np.array([10, 20, 30])
    values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dense_shape = tf.constant([40, 3])
    a = tf.IndexedSlices(values=values, indices=indices, dense_shape=dense_shape)
    mask_indices = np.array([10, 30])
    name = "mask_op_6"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    indices = np.array([7, 14, 21, 28])
    values = np.array([1.5, 2.5, 3.5, 4.5])
    dense_shape = tf.constant([30])
    a = tf.IndexedSlices(values=values, indices=indices, dense_shape=dense_shape)
    mask_indices = np.array([14, 28])
    name = "mask_op_7"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    indices = np.array([1, 2, 3])
    values = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    dense_shape = tf.constant([5, 4])
    a = tf.IndexedSlices(values=values, indices=indices, dense_shape=dense_shape)
    mask_indices = np.array([2])
    name = "mask_op_8"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    indices = np.array([0, 5, 10, 15])
    values = np.array([100, 200, 300, 400])
    dense_shape = tf.constant([20])
    a = tf.IndexedSlices(values=values, indices=indices, dense_shape=dense_shape)
    mask_indices = np.array([0, 10, 15])
    name = "mask_op_9"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    indices = np.array([3, 6, 9])
    values = np.array([[1, 1], [2, 2], [3, 3]])
    dense_shape = tf.constant([12, 2])
    a = tf.IndexedSlices(values=values, indices=indices, dense_shape=dense_shape)
    mask_indices = np.array([3, 9])
    name = "mask_op_10"
    input_dict = {"a": a, "mask_indices": mask_indices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.mask"] = tf_sparse_mask_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.mask' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.mask'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.mask', generated_inputs['tf.sparse.mask'], lib="tf", suffix=0)
