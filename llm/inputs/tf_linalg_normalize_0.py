
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_normalize_inputs():
    list_of_inputs = []

    # Input 1: Vector normalization with Euclidean norm and no axis specified
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ord = 'euclidean'
    axis = None
    name = 'normalize_vector_1'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Matrix normalization with Frobenius norm and axis specified
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    ord = 'fro'
    axis = (0, 1)
    name = 'normalize_matrix_1'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batch of vectors normalization with 1-norm and axis specified
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex64)
    ord = '1'
    axis = (1,)
    name = 'normalize_batch_1'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor normalization with inf-norm and axis specified
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.complex128)
    ord = 'inf'
    axis = (1, 2)
    name = 'normalize_tensor_1'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values and p-norm
    tensor = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    ord = 2.5
    axis = None
    name = 'normalize_vector_2'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: different axis
    tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    ord = 'euclidean'
    axis = (0,)
    name = 'normalize_matrix_2'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor, axis = 0
    tensor = np.random.rand(3, 4, 5).astype(np.float32)
    ord = 'euclidean'
    axis = (0,)
    name = 'normalize_tensor_3'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D tensor, axis = (1,2)
    tensor = np.random.rand(3, 4, 5).astype(np.float64)
    ord = 'fro'
    axis = (1, 2)
    name = 'normalize_tensor_4'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D tensor, axis = (0,1)
    tensor = np.random.rand(2, 3, 4, 5).astype(np.complex64)
    ord = 1.0
    axis = (0, 1)
    name = 'normalize_tensor_5'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D tensor, axis = (2,3)
    tensor = np.random.rand(2, 3, 4, 5).astype(np.complex128)
    ord = np.inf
    axis = (2, 3)
    name = 'normalize_tensor_6'
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
