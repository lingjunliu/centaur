
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_edit_distance_inputs():
    list_of_inputs = []

    # Input 1
    hypothesis = tf.SparseTensor(indices=[[0, 0, 0]], values=[b'a'], dense_shape=[1, 1, 1])
    truth = tf.SparseTensor(indices=[[0, 0, 0]], values=[b'a'], dense_shape=[1, 1, 1])
    normalize = True
    name = 'edit_distance_1'
    input_dict = {'hypothesis': hypothesis, 'truth': truth, 'normalize': normalize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    hypothesis = tf.SparseTensor(indices=[[0, 0, 0]], values=[b'a'], dense_shape=[1, 1, 1])
    truth = tf.SparseTensor(indices=[[0, 0, 0]], values=[b'b'], dense_shape=[1, 1, 1])
    normalize = False
    name = 'edit_distance_2'
    input_dict = {'hypothesis': hypothesis, 'truth': truth, 'normalize': normalize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    hypothesis = tf.SparseTensor(indices=[[0, 0, 0], [0, 0, 1]], values=[b'a', b'b'], dense_shape=[1, 1, 2])
    truth = tf.SparseTensor(indices=[[0, 0, 0]], values=[b'a'], dense_shape=[1, 1, 1])
    normalize = True
    name = 'edit_distance_3'
    input_dict = {'hypothesis': hypothesis, 'truth': truth, 'normalize': normalize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    hypothesis = tf.SparseTensor(indices=[[0, 0, 0]], values=[b'a'], dense_shape=[1, 1, 1])
    truth = tf.SparseTensor(indices=[[0, 0, 0], [0, 0, 1]], values=[b'a', b'b'], dense_shape=[1, 1, 2])
    normalize = False
    name = 'edit_distance_4'
    input_dict = {'hypothesis': hypothesis, 'truth': truth, 'normalize': normalize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    hypothesis = tf.SparseTensor(indices=[[0, 0, 0], [1, 0, 0]], values=[b'a', b'b'], dense_shape=[2, 1, 1])
    truth = tf.SparseTensor(indices=[[0, 0, 0], [1, 0, 0]], values=[b'a', b'b'], dense_shape=[2, 1, 1])
    normalize = True
    name = 'edit_distance_5'
    input_dict = {'hypothesis': hypothesis, 'truth': truth, 'normalize': normalize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    hypothesis = tf.SparseTensor(indices=[[0, 0, 0], [1, 0, 0]], values=[b'a', b'b'], dense_shape=[2, 1, 1])
    truth = tf.SparseTensor(indices=[[0, 0, 0], [1, 0, 0]], values=[b'c', b'd'], dense_shape=[2, 1, 1])
    normalize = False
    name = 'edit_distance_6'
    input_dict = {'hypothesis': hypothesis, 'truth': truth, 'normalize': normalize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    hypothesis = tf.SparseTensor(indices=[[0, 0, 0]], values=[b'a'], dense_shape=[1, 1, 1])
    truth = tf.SparseTensor(indices=[[0, 0, 0], [0, 1, 0]], values=[b'a', b'b'], dense_shape=[1, 2, 1])
    normalize = True
    name = 'edit_distance_7'
    input_dict = {'hypothesis': hypothesis, 'truth': truth, 'normalize': normalize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    hypothesis = tf.SparseTensor(indices=[[0, 1, 0]], values=[b'b'], dense_shape=[1, 2, 1])
    truth = tf.SparseTensor(indices=[[0, 0, 0]], values=[b'a'], dense_shape=[1, 1, 1])
    normalize = False
    name = 'edit_distance_8'
    input_dict = {'hypothesis': hypothesis, 'truth': truth, 'normalize': normalize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    hypothesis = tf.SparseTensor(indices=[[0, 0, 0], [1, 0, 0]], values=[b'a', b'b'], dense_shape=[2, 1, 1])
    truth = tf.SparseTensor(indices=[[0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0]], values=[b'a', b'b', b'c', b'a'], dense_shape=[2, 2, 2])
    normalize = True
    name = 'edit_distance_9'
    input_dict = {'hypothesis': hypothesis, 'truth': truth, 'normalize': normalize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    hypothesis = tf.SparseTensor(indices=[[0, 0, 0], [1, 0, 0]], values=[b'a', b'b'], dense_shape=[2, 1, 1])
    truth = tf.SparseTensor(indices=[[0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0]], values=[b'a', b'b', b'c', b'a'], dense_shape=[2, 2, 2])
    normalize = False
    name = 'edit_distance_10'
    input_dict = {'hypothesis': hypothesis, 'truth': truth, 'normalize': normalize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    hypothesis = tf.SparseTensor(
        indices=[[0, 0, 0],[1, 0, 0]],
        values=[b"a", b"b"],
        dense_shape=[2, 1, 1])

    truth = tf.SparseTensor(
        indices=[[0, 1, 0],[1, 0, 0],[1, 0, 1],[1, 1, 0]],
        values=[b"a", b"b", b"c", b"a"],
        dense_shape=[2, 2, 2])
    normalize = True
    name = 'edit_distance_11'
    input_dict = {'hypothesis': hypothesis, 'truth': truth, 'normalize': normalize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    hypothesis = tf.SparseTensor(indices=[[0, 0, 0]], values=[b''], dense_shape=[1, 1, 1])
    truth = tf.SparseTensor(indices=[[0, 0, 0]], values=[b'a'], dense_shape=[1, 1, 1])
    normalize = True
    name = 'edit_distance_12'
    input_dict = {'hypothesis': hypothesis, 'truth': truth, 'normalize': normalize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13
    hypothesis = tf.SparseTensor(indices=[[0, 0, 0]], values=[b'hello'], dense_shape=[1, 1, 1])
    truth = tf.SparseTensor(indices=[[0, 0, 0]], values=[b'world'], dense_shape=[1, 1, 1])
    normalize = False
    name = 'edit_distance_13'
    input_dict = {'hypothesis': hypothesis, 'truth': truth, 'normalize': normalize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14: empty hypothesis
    hypothesis = tf.SparseTensor(indices=[], values=[], dense_shape=[1, 0, 0])
    truth = tf.SparseTensor(indices=[[0, 0, 0]], values=[b'a'], dense_shape=[1, 1, 1])
    normalize = True
    name = 'edit_distance_14'
    input_dict = {'hypothesis': hypothesis, 'truth': truth, 'normalize': normalize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 15: empty truth
    hypothesis = tf.SparseTensor(indices=[[0, 0, 0]], values=[b'a'], dense_shape=[1, 1, 1])
    truth = tf.SparseTensor(indices=[], values=[], dense_shape=[1, 0, 0])
    normalize = False
    name = 'edit_distance_15'
    input_dict = {'hypothesis': hypothesis, 'truth': truth, 'normalize': normalize, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.edit_distance"] = tf_edit_distance_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.edit_distance' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.edit_distance'.")

check_valid('tf.edit_distance', generated_inputs['tf.edit_distance'], lib="tf", suffix=0)
