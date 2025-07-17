
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_reshape_inputs():
    list_of_inputs = []

    def to_tuple(x):
        if isinstance(x, np.ndarray):
            return tuple(x.tolist())
        return tuple(x)

    # Input 1
    indices = np.array([[0, 0], [1, 2]])
    values = np.array([1, 2])
    shape = np.array([3, 4])
    sp_input = tf.SparseTensor(indices, values, shape)
    shape_tensor = np.array([12])
    name = "reshape_1"
    input_dict = {"sp_input": sp_input, "shape": shape_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]])
    values = np.array([1, 2, 3])
    shape = np.array([2, 2, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    shape_tensor = np.array([4, 2])
    name = "reshape_2"
    input_dict = {"sp_input": sp_input, "shape": shape_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([[0, 0], [1, 1], [2, 2]])
    values = np.array([1, 2, 3])
    shape = np.array([3, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    shape_tensor = np.array([9])
    name = "reshape_3"
    input_dict = {"sp_input": sp_input, "shape": shape_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Using -1
    indices = np.array([[0, 0], [1, 1]])
    values = np.array([1, 2])
    shape = np.array([2, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    shape_tensor = np.array([4, -1])
    name = "reshape_4"
    input_dict = {"sp_input": sp_input, "shape": shape_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D to 2D
    indices = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    values = np.array([1, 2, 3])
    shape = np.array([2, 2, 1])
    sp_input = tf.SparseTensor(indices, values, shape)
    shape_tensor = np.array([2, 2])
    name = "reshape_5"
    input_dict = {"sp_input": sp_input, "shape": shape_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Reshape to same shape
    indices = np.array([[0, 0], [1, 1]])
    values = np.array([1, 2])
    shape = np.array([2, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    shape_tensor = np.array([2, 2])
    name = "reshape_7"
    input_dict = {"sp_input": sp_input, "shape": shape_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger shape with zeros implicit
    indices = np.array([[0, 0]])
    values = np.array([1])
    shape = np.array([2, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    shape_tensor = np.array([4])
    name = "reshape_8"
    input_dict = {"sp_input": sp_input, "shape": shape_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different data type
    indices = np.array([[0, 0], [1, 1]])
    values = np.array([1.0, 2.0])
    shape = np.array([2, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    shape_tensor = np.array([4])
    name = "reshape_9"
    input_dict = {"sp_input": sp_input, "shape": shape_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Higher dimensions
    indices = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
    values = np.array([1, 2])
    shape = np.array([2, 2, 2, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    shape_tensor = np.array([16])
    name = "reshape_10"
    input_dict = {"sp_input": sp_input, "shape": shape_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Reshape with a single element
    indices = np.array([[0, 0]])
    values = np.array([5])
    shape = np.array([1, 1])
    sp_input = tf.SparseTensor(indices, values, shape)
    shape_tensor = np.array([1])
    name = "reshape_11"
    input_dict = {"sp_input": sp_input, "shape": shape_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.reshape"] = tf_sparse_reshape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.reshape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.reshape'.")

check_valid('tf.sparse.reshape', generated_inputs['tf.sparse.reshape'], lib="tf", suffix=0)
