
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sets_union_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two SparseTensors
    a = tf.sparse.SparseTensor(indices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], values=[1, 2, 3], dense_shape=[1, 2, 3])
    b = tf.sparse.SparseTensor(indices=[[0, 0, 0], [0, 0, 2], [0, 1, 1]], values=[1, 4, 5], dense_shape=[1, 2, 3])
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Two SparseTensors with different dense shapes in the last dimension
    a = tf.sparse.SparseTensor(indices=[[0, 0, 0], [0, 0, 1]], values=[1, 2], dense_shape=[1, 2, 2])
    b = tf.sparse.SparseTensor(indices=[[0, 0, 0], [0, 0, 1], [0, 0, 2]], values=[1, 3, 4], dense_shape=[1, 2, 3])
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: One SparseTensor and one dense Tensor
    a = tf.sparse.SparseTensor(indices=[[0, 0, 0], [0, 0, 1]], values=[1, 2], dense_shape=[1, 2, 2])
    b = tf.constant([[[1, 3]]], dtype=tf.int64)
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: One dense Tensor and one SparseTensor
    a = tf.constant([[[1, 2]]], dtype=tf.int64)
    b = tf.sparse.SparseTensor(indices=[[0, 0, 0], [0, 0, 1]], values=[1, 3], dense_shape=[1, 1, 2])
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Two dense Tensors
    a = tf.constant([[[1, 2], [3, 4]]], dtype=tf.int64)
    b = tf.constant([[[1, 3], [4, 5]]], dtype=tf.int64)
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Two SparseTensors.
    a = tf.sparse.SparseTensor(indices=[[0, 0, 0]], values=[1], dense_shape=[1, 1, 1])
    b = tf.sparse.SparseTensor(indices=[[0, 0, 0]], values=[2], dense_shape=[1, 1, 1])
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Two Tensors with shape [2, 2, 1]
    a = tf.constant([[[1], [2]], [[3], [4]]], dtype=tf.int64)
    b = tf.constant([[[2], [3]], [[4], [5]]], dtype=tf.int64)
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: validate_indices = False
    a = tf.constant([[[1, 2, 3], [4, 5, 6]]], dtype=tf.int64)
    b = tf.constant([[[2, 4, 5], [5, 7, 8]]], dtype=tf.int64)
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: One tensor of shape [1,2,1,2] and another one of shape [1,2,1,3]
    a = tf.constant([[[[1, 2]],[[3,4]]]], dtype=tf.int64)
    b = tf.constant([[[[2, 3, 5]],[[5, 7, 9]]]], dtype=tf.int64)
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: One sparse tensor of shape [1, 1, 2] and another one of shape [1, 1, 3]
    a = tf.sparse.SparseTensor(indices=[[0, 0, 0], [0, 0, 1]], values=[1, 2], dense_shape=[1, 1, 2])
    b = tf.sparse.SparseTensor(indices=[[0, 0, 0], [0, 0, 1], [0, 0, 2]], values=[1, 3, 5], dense_shape=[1, 1, 3])
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Convert SparseTensor to numpy array for abstract input generation
    for input_dict in list_of_inputs:
        if isinstance(input_dict["a"], tf.sparse.SparseTensor):
            input_dict["a"] = tf.sparse.to_dense(input_dict["a"])
        if isinstance(input_dict["b"], tf.sparse.SparseTensor):
            input_dict["b"] = tf.sparse.to_dense(input_dict["b"])

        if isinstance(input_dict["a"], tf.Tensor):
            input_dict["a"] = input_dict["a"].numpy()
        if isinstance(input_dict["b"], tf.Tensor):
            input_dict["b"] = input_dict["b"].numpy()

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sets.union"] = tf_sets_union_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sets.union' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sets.union'.")

check_valid('tf.sets.union', generated_inputs['tf.sets.union'], lib="tf", suffix=0)
