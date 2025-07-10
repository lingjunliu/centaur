
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sets_size_inputs():
    list_of_inputs = []

    def create_sparse_tensor(indices, values, dense_shape):
        return tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    # Input 1: Basic case with valid indices
    indices = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
    values = np.array([1, 2, 1, 3, 4, 3])
    dense_shape = np.array([2, 5])
    a = create_sparse_tensor(indices, values, dense_shape)
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: validate_indices=False
    indices = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
    values = np.array([1, 2, 1, 3, 4, 3])
    dense_shape = np.array([2, 5])
    a = create_sparse_tensor(indices, values, dense_shape)
    validate_indices = False
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty SparseTensor
    indices = np.array([]).reshape(0, 2)
    values = np.array([])
    dense_shape = np.array([2, 5])
    a = create_sparse_tensor(indices, values, dense_shape)
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Higher rank SparseTensor
    indices = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]])
    values = np.array([1, 2, 3, 4])
    dense_shape = np.array([1, 2, 3])
    a = create_sparse_tensor(indices, values, dense_shape)
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different values
    indices = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
    values = np.array([5, 6, 7, 8, 9, 10])
    dense_shape = np.array([2, 5])
    a = create_sparse_tensor(indices, values, dense_shape)
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single row SparseTensor
    indices = np.array([[0, 0], [0, 1], [0, 2]])
    values = np.array([1, 2, 3])
    dense_shape = np.array([1, 5])
    a = create_sparse_tensor(indices, values, dense_shape)
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single column SparseTensor
    indices = np.array([[0, 0], [1, 0], [2, 0]])
    values = np.array([1, 2, 3])
    dense_shape = np.array([5, 1])
    a = create_sparse_tensor(indices, values, dense_shape)
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: With zero values
    indices = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
    values = np.array([0, 1, 0, 2, 3, 0])
    dense_shape = np.array([2, 5])
    a = create_sparse_tensor(indices, values, dense_shape)
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: SparseTensor with all same values
    indices = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
    values = np.array([1, 1, 1, 1, 1, 1])
    dense_shape = np.array([2, 5])
    a = create_sparse_tensor(indices, values, dense_shape)
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Sparse tensor with a single value
    indices = np.array([[0, 0]])
    values = np.array([5])
    dense_shape = np.array([1, 1])
    a = create_sparse_tensor(indices, values, dense_shape)
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Sparse tensor with a different dense shape and values
    indices = np.array([[0, 0], [1, 2], [2, 1]])
    values = np.array([7, 8, 9])
    dense_shape = np.array([3, 4])
    a = create_sparse_tensor(indices, values, dense_shape)
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13: Sparse tensor with unsorted indices and validate_indices=False
    indices = np.array([[1, 0], [0, 1], [0, 0]])
    values = np.array([4, 2, 1])
    dense_shape = np.array([2, 2])
    a = create_sparse_tensor(indices, values, dense_shape)
    validate_indices = False
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14: All zeros SparseTensor
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values = np.array([0, 0, 0, 0])
    dense_shape = np.array([2, 2])
    a = create_sparse_tensor(indices, values, dense_shape)
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 15: Sparse tensor with very large dense shape
    indices = np.array([[0, 0], [9999, 9999]])
    values = np.array([1, 2])
    dense_shape = np.array([10000, 10000])
    a = create_sparse_tensor(indices, values, dense_shape)
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sets.size"] = tf_sets_size_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sets.size' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sets.size'.")

check_valid('tf.sets.size', generated_inputs['tf.sets.size'], lib="tf", suffix=0)
