
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_reset_shape_inputs():
    list_of_inputs = []

    def get_range(tensor):
      if isinstance(tensor, tf.SparseTensor):
        values = tensor.values
        if values.shape.rank == 0:
          return [values.numpy(), values.numpy()] if values.numpy() is not None else [0,0]
        else:
          return [np.min(values.numpy()), np.max(values.numpy())] if values.numpy().size > 0 else [0, 0]
      else:
        return [np.min(tensor), np.max(tensor)] if tensor.size > 0 else [0, 0]

    # Input 1
    indices = np.array([[0, 0], [1, 2]])
    values = np.array([1, 2])
    shape = np.array([3, 4])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([4, 5])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 2]])
    values = np.array([1, 2, 3])
    shape = np.array([2, 2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([3, 3, 4])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = np.array([[0, 0], [1, 1]])
    values = np.array([1, 2])
    shape = np.array([2, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = None
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = np.array([[0, 0, 0], [1, 2, 1]])
    values = np.array([1, 2])
    shape = np.array([2, 3, 4])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([3, 4, 5])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices = np.array([[0, 0, 0, 0]])
    values = np.array([1])
    shape = np.array([1, 1, 1, 1])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([2, 2, 2, 2])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices = np.array([[0, 1]])
    values = np.array([2])
    shape = np.array([1, 2])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([5, 5])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    indices = np.array([[0, 0, 0]])
    values = np.array([1])
    shape = np.array([1, 1, 1])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([1, 1, 2])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    indices = np.array([[0, 0]])
    values = np.array([1])
    shape = np.array([1, 1])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([2, 2])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    indices = np.array([[1, 2, 3]])
    values = np.array([4])
    shape = np.array([5, 6, 7])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = None
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    indices = np.array([[0, 0], [1, 2]])
    values = np.array([1, 2])
    shape = np.array([3, 4])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([5, 4])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 - Corrected indices for empty SparseTensor
    indices = np.empty((0, 2), dtype=np.int64)
    values = np.array([])
    shape = np.array([2, 3])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([3,4])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12 - All zeros
    indices = np.array([[0, 0]])
    values = np.array([0])
    shape = np.array([1, 1])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([2, 2])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13 - Negative values in indices
    indices = np.array([[0, 0]])
    values = np.array([-1])
    shape = np.array([1, 1])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([2, 2])
    input_dict = {"sp_input": sp_input, "new_shape": new_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14 - float values
    indices = np.array([[0, 0]])
    values = np.array([1.5])
    shape = np.array([1, 1])
    sp_input = tf.SparseTensor(indices, values, shape)
    new_shape = np.array([2, 2])
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
