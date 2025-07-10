
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_einsum_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    input_dict = {"subscripts": "i,i->", "operands": [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    input_dict = {"subscripts": "ij,ij->ij", "operands": [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    input_dict = {"subscripts": "ij,j->i", "operands": [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    input_dict = {"subscripts": "ij,ji->i", "operands": [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    input_dict = {"subscripts": "ijk,ijk->ijk", "operands": [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    input_dict = {"subscripts": "ijk,lmn->", "operands": [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    input_dict = {"subscripts": "ij,j->ij", "operands": [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.array([1, 2, 3])
    b = np.array([[4, 5, 6], [7,8,9]])
    input_dict = {"subscripts": "i,ij->j", "operands": [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.array([[-1, 2], [3, -4]])
    b = np.array([[5, -6], [-7, 8]])
    input_dict = {"subscripts": "ij,ji->", "operands": [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    a = np.random.rand(2,3,4)
    b = np.random.rand(4,5)
    input_dict = {"subscripts": "ijk,kl->ijl", "operands": [tf.convert_to_tensor(a), tf.convert_to_tensor(b)]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.einsum"] = tf_experimental_numpy_einsum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.einsum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.einsum'.")

check_valid('tf.experimental.numpy.einsum', generated_inputs['tf.experimental.numpy.einsum'], lib="tf", suffix=0)
