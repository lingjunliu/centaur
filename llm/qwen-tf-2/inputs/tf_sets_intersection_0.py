
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sets_intersection_inputs():
    list_of_inputs = []
    
    # Input 1, valid - 2D tensors
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[2, 4, 5], [3, 6, 7]])
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid - 3D tensors
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]])
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid - 1D tensors
    a = np.array([1, 2, 3, 4])
    b = np.array([2, 3, 4, 5])
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid - 4D tensors
    a = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[9, 10], [11, 12]]])
    b = np.array([[[[2, 3], [4, 5]], [[6, 7], [8, 9]]], [[10, 11], [12, 13]]])
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid - negative values
    a = np.array([[1, -2, 3], [-4, 5, 6]])
    b = np.array([[2, -4, -6], [5, 7, 9]])
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - sparse tensors with different shapes
    a = tf.sparse.SSparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1, 2, 3], dense_shape=[2, 2])
    b = tf.sparse.SSparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1, 2, 3], dense_shape=[2, 2])
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - sparse tensors with same shape but different indices
    a = tf.sparse.SSparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1, 2, 3], dense_shape=[2, 2])
    b = tf.sparse.SSparseTensor(indices=[[0, 0], [0, 1], [1, 1]], values=[1, 2, 4], dense_shape=[2, 2])
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - sparse tensors with different shapes
    a = tf.sparse.SSparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1, 2, 3], dense_shape=[2, 2])
    b = tf.sparse.SSparseTensor(indices=[[0, 0], [0, 1], [1, 0], [1, 1]], values=[1, 2, 3, 4], dense_shape=[2, 3])
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - mixed types (int and float)
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[2.0, 3.0], [4.0, 5.0]])
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - empty arrays
    a = np.array([[], []])
    b = np.array([[], []])
    validate_indices = True
    
    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.sets.intersection"] = tf_sets_intersection_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sets.intersection' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sets.intersection'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sets.intersection', generated_inputs['tf.sets.intersection'], lib="tf", suffix=0)
