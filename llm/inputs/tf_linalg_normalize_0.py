
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_normalize_inputs():
    list_of_inputs = []

    # Input 1: Basic vector normalization
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ord = 'euclidean'
    axis = None
    name = 'norm1'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Matrix Frobenius norm normalization
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'fro'
    axis = None
    name = 'norm2'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batch of vectors, normalize along axis 1
    tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    ord = 'euclidean'
    axis = (1,)
    name = 'norm3'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch of matrices, normalize along axes (0, 1) using 1-norm
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    ord = '1'
    axis = (0, 1)
    name = 'norm4'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative axis for vector normalization
    tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    ord = 'euclidean'
    axis = (-1,)
    name = 'norm5'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: np.inf norm
    tensor = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    ord = np.inf
    axis = None
    name = 'norm6'
    input_dict = {'tensor': tensor, 'ord': str(ord), 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: p-norm with p = 1.5
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ord = '1.5'
    axis = None
    name = 'norm7'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex tensor, euclidean norm
    tensor = np.array([1 + 1j, 2 - 2j, 3 + 0j], dtype=np.complex64)
    ord = 'euclidean'
    axis = None
    name = 'norm8'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Higher dimensional tensor, axis = (1,)
    tensor = np.random.rand(2, 3, 4).astype(np.float32)
    ord = 'euclidean'
    axis = (1,)
    name = 'norm9'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Double type tensor
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    ord = 'euclidean'
    axis = None
    name = 'norm11'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14: Different order
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ord = '1'
    axis = None
    name = 'norm14'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 15: Axis as tuple with one element.
    tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    ord = 'euclidean'
    axis = (0,)
    name = 'norm15'
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
