
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_norm_inputs():
    list_of_inputs = []

    # Input 1: Vector, default norm
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ord = 'euclidean'
    axis = None
    keepdims = False
    name = 'norm_1'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Matrix, Frobenius norm
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'fro'
    axis = None
    keepdims = False
    name = 'norm_2'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Matrix, 1-norm
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = '1'
    axis = None
    keepdims = False
    name = 'norm_3'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Matrix, inf-norm
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = np.inf
    axis = None
    keepdims = False
    name = 'norm_4'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batch of vectors, axis=1
    tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    ord = 'euclidean'
    axis = (1,)
    keepdims = False
    name = 'norm_5'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batch of matrices, axis=(0, 1), keepdims=True
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    ord = 'fro'
    axis = (0, 1)
    keepdims = True
    name = 'norm_6'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Vector, ord=2
    tensor = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    ord = '2'
    axis = None
    keepdims = False
    name = 'norm_7'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D tensor, axis=(1,2)
    tensor = np.random.rand(3, 4, 5).astype(np.float32)
    ord = 'fro'
    axis = (1,2)
    keepdims = False
    name = 'norm_8'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex tensor with axis
    tensor = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    ord = 'euclidean'
    axis = (0,)
    keepdims = False
    name = 'norm_9'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Matrix norm, axis = (-2, -1)
    tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    ord = 'fro'
    axis = (-2, -1)
    keepdims = False
    name = 'norm_10'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.norm"] = tf_norm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.norm'.")

check_valid('tf.norm', generated_inputs['tf.norm'], lib="tf", suffix=0)
