
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_reset_shape_inputs():
    list_of_inputs = []

    # Input 1: Basic SparseTensor with new_shape
    indices = np.array([[0, 0], [1, 2]])
    values = np.array([1, 2])
    shape = np.array([2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([3, 4])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: SparseTensor with None new_shape (tight bounding box)
    indices = np.array([[0, 1], [2, 0]])
    values = np.array([3, 4])
    shape = np.array([3, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = None
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty SparseTensor with new_shape
    indices = np.array([]).reshape(0, 2)
    values = np.array([])
    shape = np.array([2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([4, 5])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty SparseTensor with None new_shape
    indices = np.array([]).reshape(0, 2)
    values = np.array([])
    shape = np.array([2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = None
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D SparseTensor with new_shape
    indices = np.array([[0, 0, 1], [1, 2, 0]])
    values = np.array([5, 6])
    shape = np.array([2, 3, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([3, 4, 3])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D SparseTensor with None new_shape
    indices = np.array([[0, 0, 1], [1, 2, 0]])
    values = np.array([5, 6])
    shape = np.array([2, 3, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = None
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Higher values in indices with new_shape
    indices = np.array([[1, 1], [2, 0]])
    values = np.array([7, 8])
    shape = np.array([3, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([4, 3])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different data type for values with new_shape
    indices = np.array([[0, 0], [1, 2]])
    values = np.array([1.0, 2.0])
    shape = np.array([2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([3, 4])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different data type for values with None new_shape
    indices = np.array([[0, 1], [2, 0]])
    values = np.array([3.0, 4.0])
    shape = np.array([3, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = None
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: SparseTensor with identity new_shape
    indices = np.array([[0, 0], [1, 2]])
    values = np.array([1, 2])
    shape = np.array([2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([2, 3])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.reset_shape"] = tf_sparse_reset_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.reset_shape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.reset_shape'.")

check_valid('tf.sparse.reset_shape', generated_inputs['tf.sparse.reset_shape'], lib="tf", suffix=0)
