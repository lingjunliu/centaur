
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_cross_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two dense tensors
    inp_0 = np.array([['a'], ['b']])
    inp_1 = np.array([['c'], ['d']])
    input_dict = {'inputs': [inp_0, inp_1], 'name': None, 'separator': '_X_'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Two dense tensors with different separators
    inp_0 = np.array([['a'], ['b']])
    inp_1 = np.array([['c'], ['d']])
    input_dict = {'inputs': [inp_0, inp_1], 'name': 'cross_op', 'separator': '_Y_'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Three dense tensors
    inp_0 = np.array([['a'], ['b']])
    inp_1 = np.array([['c'], ['d']])
    inp_2 = np.array([['e'], ['f']])
    input_dict = {'inputs': [inp_0, inp_1, inp_2], 'name': None, 'separator': '_X_'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Two dense tensors, different shapes
    inp_0 = np.array([['a', 'b'], ['c', 'd']])
    inp_1 = np.array([['e'], ['f']])
    input_dict = {'inputs': [inp_0, inp_1], 'name': None, 'separator': '_X_'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  One dense tensor
    inp_0 = np.array([['a'], ['b']])
    input_dict = {'inputs': [inp_0], 'name': None, 'separator': '_X_'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Dense tensors with numeric values as strings
    inp_0 = np.array([['1'], ['2']])
    inp_1 = np.array([['3'], ['4']])
    input_dict = {'inputs': [inp_0, inp_1], 'name': None, 'separator': '_X_'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Dense tensor with empty string
    inp_0 = np.array([[''], ['b']])
    inp_1 = np.array([['c'], ['d']])
    input_dict = {'inputs': [inp_0, inp_1], 'name': None, 'separator': '_X_'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Valid Case, shape mismatch fixed
    inp_0 = np.array([['a', 'b', 'c'], ['d', 'e', 'f']])
    inp_1 = np.array([['g', 'h', 'i'], ['j', 'k', 'l']])
    input_dict = {'inputs': [inp_0, inp_1], 'name': None, 'separator': '_X_'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Valid case
    inp_0 = np.array([['a'], ['b']])
    inp_1 = np.array([['c'], ['d']]) #Keep the shapes consistent
    input_dict = {'inputs': [inp_0, inp_1], 'name': None, 'separator': '_X_'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Valid Case, Shape mismatch fixed
    inp_0 = np.array([['a', 'b'], ['c', 'd']])
    inp_1 = np.array([['e', 'f'], ['g', 'h']])
    input_dict = {'inputs': [inp_0, inp_1], 'name': None, 'separator': '_X_'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.cross"] = tf_sparse_cross_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.cross'.")

check_valid('tf.sparse.cross', generated_inputs['tf.sparse.cross'], lib="tf", suffix=0)
