
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic vector norm (euclidean)
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ord = 'euclidean'
    axis = None
    keepdims = False
    name = None
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Matrix Frobenius norm
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'fro'
    axis = None
    keepdims = False
    name = None
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Vector 1-norm
    tensor = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    ord = '1'
    axis = None
    keepdims = False
    name = None
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Vector inf-norm
    tensor = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    ord = np.inf
    axis = None
    keepdims = False
    name = None
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Axis specified for batch of vectors
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'euclidean'
    axis = [1]
    keepdims = False
    name = None
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Axis specified, keepdims=True
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'euclidean'
    axis = [1]
    keepdims = True
    name = None
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Matrix 1-norm (induced)
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = '1'
    axis = [0, 1]
    keepdims = False
    name = None
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Matrix inf-norm (induced)
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = np.inf
    axis = [0, 1]
    keepdims = False
    name = None
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with negative axis
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    ord = 'euclidean'
    axis = [-1]
    keepdims = False
    name = None
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex Tensor
    tensor = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    ord = 'euclidean'
    axis = None
    keepdims = False
    name = None
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.norm_2"] = tf_norm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.norm_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.norm_2'.")

check_valid('tf.norm', generated_inputs['tf.norm_2'], lib="tf", suffix=2)
