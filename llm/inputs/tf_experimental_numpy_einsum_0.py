
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_einsum_inputs():
    list_of_inputs = []

    # Input 1
    subscripts = "ij,jk->ik"
    operands = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    input_dict = {"subscripts": subscripts, "operands": [operands[0].tolist(), operands[1].tolist()]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    subscripts = "i->i"
    operands = [np.array([1, 2, 3])]
    input_dict = {"subscripts": subscripts, "operands": [operands[0].tolist()]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    subscripts = "...,ij->..."
    operands = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[9, 10], [11, 12]])]
    input_dict = {"subscripts": subscripts, "operands": [operands[0].tolist(), operands[1].tolist()]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    subscripts = "i,i->"
    operands = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    input_dict = {"subscripts": subscripts, "operands": [operands[0].tolist(), operands[1].tolist()]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    subscripts = "ij,i->j"
    operands = [np.array([[1, 2], [3, 4], [5, 6]]), np.array([7, 8, 9])]
    input_dict = {"subscripts": subscripts, "operands": [operands[0].tolist(), operands[1].tolist()]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    subscripts = "i,j,k->ijk"
    operands = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    input_dict = {"subscripts": subscripts, "operands": [operands[0].tolist(), operands[1].tolist(), operands[2].tolist()]}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    subscripts = "ii->i"
    operands = [np.array([[1, 2], [3, 4]])]
    input_dict = {"subscripts": subscripts, "operands": [operands[0].tolist()]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    subscripts = "ij,kl,mn->imn"
    operands = [np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]]), np.array([[9,10],[11,12]])]
    input_dict = {"subscripts": subscripts, "operands": [operands[0].tolist(), operands[1].tolist(), operands[2].tolist()]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    subscripts = "i,j->ij"
    operands = [np.array([1, 2, 3]), np.array([4, 5])]
    input_dict = {"subscripts": subscripts, "operands": [operands[0].tolist(), operands[1].tolist()]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    subscripts = "..."
    operands = [np.array([1, 2, 3])]
    input_dict = {"subscripts": subscripts, "operands": [operands[0].tolist()]}
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
