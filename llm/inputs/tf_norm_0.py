
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
    name = 'norm_vector'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Matrix, Frobenius norm
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'fro'
    axis = None
    keepdims = False
    name = 'norm_matrix_fro'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batch of vectors, 1-norm along axis 1
    tensor = np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32)
    ord = '1'
    axis = (1,)
    keepdims = True
    name = 'norm_batch_vector_1'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch of matrices, inf-norm along axis (0, 1)
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    ord = np.inf
    axis = (1, 2)
    keepdims = False
    name = 'norm_batch_matrix_inf'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Vector, 2-norm
    tensor = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    ord = '2'
    axis = None
    keepdims = False
    name = 'norm_vector_2'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Matrix, 1-norm (induced)
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = '1'
    axis = (0, 1)
    keepdims = False
    name = 'norm_matrix_1'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor, euclidean norm (treat as vector)
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    ord = 'euclidean'
    axis = None
    keepdims = False
    name = 'norm_tensor_euclidean'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Batch of vectors, default norm along axis 0, keepdims
    tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    ord = 'euclidean'
    axis = (0,)
    keepdims = True
    name = 'norm_batch_vector_default_keepdims'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex tensor
    tensor = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    ord = 'fro'
    axis = None
    keepdims = False
    name = 'complex_tensor'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: higher dimension tensor, explicit axis = None
    tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    ord = 'euclidean'
    axis = None
    keepdims = False
    name = 'higher_dim_tensor'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: vector with p-norm
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ord = '1.5'
    axis = None
    keepdims = False
    name = 'vector_pnorm'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: negative axis
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'euclidean'
    axis = (-1,)
    keepdims = False
    name = 'negative_axis'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: keepdims True with axis
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'euclidean'
    axis = (0,)
    keepdims = True
    name = 'keepdims_true'
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
