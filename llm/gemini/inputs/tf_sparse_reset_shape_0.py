
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_reset_shape_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D SparseTensor with new_shape
    indices = np.array([[0, 0], [1, 2]])
    values = np.array([1, 2])
    shape = np.array([2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([2, 4])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple 2D SparseTensor with None new_shape
    indices = np.array([[0, 0], [1, 2]])
    values = np.array([1, 2])
    shape = np.array([2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = None
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D SparseTensor with new_shape
    indices = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 2]])
    values = np.array([1, 2, 3])
    shape = np.array([2, 2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([2, 2, 4])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D SparseTensor with None new_shape
    indices = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 2]])
    values = np.array([1, 2, 3])
    shape = np.array([2, 2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = None
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D SparseTensor with zeros and new_shape
    indices = np.array([[0, 0], [1, 2], [1, 0]])
    values = np.array([1, 0, 2])
    shape = np.array([2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([3, 3])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty SparseTensor with new_shape
    indices = np.array([]).reshape(0, 2)
    values = np.array([])
    shape = np.array([2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([3, 4])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty SparseTensor with None new_shape
    indices = np.array([]).reshape(0, 2)
    values = np.array([])
    shape = np.array([2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = None
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D SparseTensor with new_shape
    indices = np.array([[0, 0, 1, 1], [0, 1, 0, 0], [1, 0, 2, 1]])
    values = np.array([1, 2, 3])
    shape = np.array([2, 2, 3, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([2, 2, 3, 3])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D SparseTensor with new_shape
    indices = np.array([[0], [2]])
    values = np.array([1, 2])
    shape = np.array([5])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([6])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D SparseTensor with None new_shape
    indices = np.array([[0], [2]])
    values = np.array([1, 2])
    shape = np.array([5])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = None
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D SparseTensor with larger new_shape
    indices = np.array([[0, 0], [1, 2]])
    values = np.array([1, 2])
    shape = np.array([2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([3, 5])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.reset_shape"] = tf_sparse_reset_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.reset_shape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.reset_shape'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.reset_shape', generated_inputs['tf.sparse.reset_shape'], lib="tf", suffix=0)
