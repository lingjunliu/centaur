
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_normalize_inputs():
    list_of_inputs = []

    # Input 1
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'euclidean'
    axis = (0, 1)
    name = 'normalize_matrix'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    ord = '1'
    axis = None
    name = 'normalize_vector_1norm'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex64)
    ord = 'fro'
    axis = (0, 1)
    name = 'normalize_complex_matrix'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tensor = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    ord = 'inf'
    axis = (0, 1)
    name = 'normalize_inf_matrix'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ord = '2'
    axis = None
    name = 'normalize_vector_2norm'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    ord = 'euclidean'
    axis = (1,2)
    name = 'normalize_batch_matrix'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ord = 'euclidean'
    axis = None
    name = 'normalize_vector_default'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = '1'
    axis = (0, 1)
    name = 'normalize_matrix_1norm'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ord = np.inf
    axis = None
    name = 'normalize_inf_vector'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    ord = '2'
    axis = (-2, -1)
    name = 'normalize_batch_matrix_2norm'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    tensor = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    ord = 'euclidean'
    axis = None
    name = 'normalize_vector_default'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    ord = 'euclidean'
    axis = (1,)
    name = 'normalize_matrix_axis1'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    ord = 'euclidean'
    axis = (2,)
    name = 'normalize_batch_matrix_axis2'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.normalize"] = tf_linalg_normalize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.normalize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.normalize'.")

check_valid('tf.linalg.normalize', generated_inputs['tf.linalg.normalize'], lib="tf", suffix=0)
