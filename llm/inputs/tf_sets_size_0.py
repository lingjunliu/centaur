
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sets_size_inputs():
    list_of_inputs = []

    # Input 1: Basic case with validate_indices=True
    a = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [1, 1]], values=[1, 2, 3, 1], dense_shape=[2, 2])
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic case with validate_indices=False
    a = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [1, 1]], values=[1, 2, 3, 1], dense_shape=[2, 2])
    validate_indices = False
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty SparseTensor - fixed
    a = tf.SparseTensor(indices=np.empty(shape=[0, 2], dtype=np.int64), values=np.array([], dtype=np.int32), dense_shape=[0, 0])
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: SparseTensor with a single value
    a = tf.SparseTensor(indices=[[0, 0]], values=[5], dense_shape=[1, 1])
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: SparseTensor with all identical values
    a = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [1, 1]], values=[7, 7, 7, 7], dense_shape=[2, 2])
    validate_indices = True
    input_dict = {"a": a, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: SparseTensor with zero as a value
    a = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [1, 1]], values=[0, 1, 0, 2], dense_shape=[2, 2])
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
