
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sets_size_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    a = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [0, 2]], values=[1, 2, 3], dense_shape=[1, 3])
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: validate_indices = False
    a = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [0, 2]], values=[1, 2, 3], dense_shape=[1, 3])
    validate_indices = False
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D SparseTensor
    a = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [1, 1]], values=[1, 2, 3, 4], dense_shape=[2, 2])
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D SparseTensor
    a = tf.sparse.SparseTensor(indices=[[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], values=[1, 2, 3, 4], dense_shape=[2, 2, 2])
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Duplicate values in the last dimension
    a = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [0, 2]], values=[1, 2, 2], dense_shape=[1, 3])
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All same values
    a = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [0, 2]], values=[1, 1, 1], dense_shape=[1, 3])
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: Larger values
    a = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [0, 2]], values=[100, 200, 300], dense_shape=[1, 3])
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 9: More rows
    a = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]], values=[1, 2, 3, 4, 5, 6], dense_shape=[3, 2])
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty sparse tensor. Changed dense_shape, and ensured values/indices are also empty.
    a = tf.sparse.SparseTensor(indices=np.array([]).reshape(0,2), values=np.array([], dtype=np.int32), dense_shape=[1, 1])
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Values are not sorted in the last dimension. validate_indices = True
    a = tf.sparse.SparseTensor(indices=[[0, 0], [0, 2], [0, 1]], values=[1, 3, 2], dense_shape=[1, 3])
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: More complex 3D
    a = tf.sparse.SparseTensor(indices=[[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], values=[1, 2, 3, 4], dense_shape=[2, 2, 2])
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Non-unique indices with validate_indices=False
    a = tf.sparse.SparseTensor(indices=[[0, 0], [0, 0], [0, 1]], values=[1, 2, 3], dense_shape=[1, 2])
    validate_indices = False
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14: Non-unique indices with validate_indices=True
    a = tf.sparse.SparseTensor(indices=[[0, 0], [0, 0], [0, 1]], values=[1, 2, 3], dense_shape=[1, 2])
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 15: Indices out of order with validate_indices=False
    a = tf.sparse.SparseTensor(indices=[[0, 1], [0, 0]], values=[3, 1], dense_shape=[1, 2])
    validate_indices = False
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
