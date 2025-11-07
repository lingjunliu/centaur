
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sets_union_inputs():
    list_of_inputs = []
    
    # Input 1: Basic dense tensor union
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    b = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.int32)
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Sparse tensor union with different shapes
    a = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1, 2, 3], dense_shape=[2, 3]
    b = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [1, 1]], values=[1, 3, 4, 5], dense_shape=[2, 4]
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Dense tensor with negative values
    a = np.array([[-1, -2], [-3, -4]], dtype=np.int32)
    b = np.array([[1, 2], [3, 4]], dtype=np.int32)
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Sparse tensor with different dimensions
    a = tf.sparse.SparseTensor(indices=[[0, 0, 0], [0, 1, 0], [1, 0, 0]], values=[1, 2, 3], dense_shape=[2, 2, 2]
    b = tf.sparse.SparseTensor(indices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], values=[1, 3, 4, 5], dense_shape=[2, 2, 3]
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Mixed sparse and dense tensor union
    a = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1, 2, 3], dense_shape=[2, 3]
    b = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Sparse tensor with single element sets
    a = tf.sparse.SparseTensor(indices=[[0, 0], [1, 0]], values=[1, 3], dense_shape=[2, 2]
    b = tf.sparse.SparseTensor(indices=[[0, 0], [1, 0]], values=[2, 4], dense_shape=[2, 2]
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Dense tensor with different number of elements in last dimension
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    b = np.array([[9, 10], [11, 12]], dtype=np.int32)
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Sparse tensor with non-unique elements
    a = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1, 2, 3], dense_shape=[2, 3]
    b = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1, 2, 3], dense_shape=[2, 3]
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Sparse tensor with large number of elements
    a = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [1, 1]], values=[1, 2, 3, 4], dense_shape=[2, 4]
    b = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [1, 1]], values=[5, 6, 7, 8], dense_shape=[2, 4]
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Sparse tensor with different data types (same type)
    a = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1, 2, 3], dense_shape=[2, 3]
    b = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1, 2, 3], dense_shape=[2, 3]
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.sets.union"] = tf_sets_union_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sets.union' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sets.union'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sets.union', generated_inputs['tf.sets.union'], lib="tf", suffix=0)
